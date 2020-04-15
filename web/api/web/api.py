import stripe
from django.contrib.auth import get_user_model
from django.contrib.postgres.search import TrigramDistance
from django.core.exceptions import SuspiciousOperation
from django.db import IntegrityError
from django.db.models import Q
from django.http import HttpResponse
from rest_framework import mixins, status, viewsets
from rest_framework.pagination import PageNumberPagination
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.response import Response

from web.models import Agent, AgentLike, Contributor, Payment, PaymentIntent, UserProfile
from web.serializers import AgentLikeSerializer, AgentRetrieveSerializer, AgentSerializer, ContributorSerializer, \
    PaymentIntentSerializer, PaymentSerializer, UserProfileSerializer, UserRetrieveSerializer
from web.settings import STRIPE_API_KEY, STRIPE_WEBHOOK_SECRET

stripe.api_key = STRIPE_API_KEY


class StandardResultsSetPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 1000


class AgentViewSet(viewsets.ModelViewSet):
    queryset = Agent.objects.all()
    serializer_class = AgentSerializer
    parser_classes = [MultiPartParser, JSONParser]
    pagination_class = StandardResultsSetPagination

    def get_queryset(self):
        if self.request.user.is_authenticated:
            qs = Q(author=self.request.user)
            if self.action in ('partial_update', 'update'):
                qs |= Q(contributors__user=self.request.user)
            if self.action in ('retrieve', 'list'):
                qs |= Q(contributors__user=self.request.user)
                qs |= Q(public=True)
        else:
            if self.action not in ('retrieve', 'list'):
                raise SuspiciousOperation('Unauthenticated users may only view public resources')
            qs = Q(public=True)

        if self.action == 'list' and 'search' in self.request.query_params:
            search = self.request.query_params['search']
            queryset = self.queryset.annotate(
                name_distance=TrigramDistance('name', search),
                description_distance=TrigramDistance('description', search),
            )
            queryset = queryset.filter(
                (Q(name_distance__lte=0.99) |
                 Q(description_distance__lte=0.9)) &
                qs
            )
            queryset = queryset.order_by('name_distance', 'description_distance')
            return queryset.distinct()
        return self.queryset.filter(qs).distinct().order_by('id')

    def get_serializer_class(self, *args, **kwargs):
        if self.action == 'retrieve':
            return AgentRetrieveSerializer
        return super(AgentViewSet, self).get_serializer_class()

    def perform_create(self, serializer):
        if not self.request.user.is_authenticated:
            raise SuspiciousOperation('Unauthenticated users may not create agents')
        try:
            serializer.save(author=serializer.context['request'].user)
        except IntegrityError:
            raise SuspiciousOperation('You already have an agent by that name associated with your account')

    def perform_update(self, serializer):
        if 'public' in self.request.data and serializer.instance.author_id != self.request.user.id:
            raise SuspiciousOperation('Only agent authors may change the publication status')
        try:
            serializer.save()
        except IntegrityError:
            raise SuspiciousOperation('You already have an agent by that name associated with your account')


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = get_user_model().objects.all()
    serializer_class = UserRetrieveSerializer
    pagination_class = StandardResultsSetPagination

    def get_queryset(self):
        if self.action == 'list' and 'search' in self.request.query_params:
            search = self.request.query_params['search']
            queryset = self.queryset.annotate(
                username_distance=TrigramDistance('username', search),
            )
            queryset = queryset.filter((Q(username_distance__lte=0.99)))
            queryset = queryset.order_by('username_distance')
            return queryset
        return self.queryset


class ContributorViewSet(mixins.CreateModelMixin,
                         mixins.RetrieveModelMixin,
                         mixins.DestroyModelMixin,
                         mixins.ListModelMixin,
                         viewsets.GenericViewSet):
    queryset = Contributor.objects.all()
    serializer_class = ContributorSerializer
    pagination_class = StandardResultsSetPagination

    def get_queryset(self):
        if self.request.user.is_authenticated:
            qs = Q(agent__author=self.request.user)
            if self.action == 'destroy':
                qs |= Q(user=self.request.user)
            if self.action in ('retrieve', 'list'):
                qs |= Q(user=self.request.user)
                qs |= Q(agent__contributors__user=self.request.user)
                qs |= Q(agent__public=True)
        else:
            if self.action not in ('retrieve', 'list'):
                raise SuspiciousOperation('Unauthenticated users may only view public resources')
            qs = Q(agent__public=True)

        if 'agent_id' in self.request.query_params:
            qs &= Q(agent_id=self.request.query_params['agent_id'])

        return self.queryset.filter(qs).distinct().order_by('id')

    def perform_create(self, serializer):
        if not self.request.user.is_authenticated:
            raise SuspiciousOperation('Unauthenticated users may not add contributors')
        agent = Agent.objects.all().filter(id=self.request.data['agent_id']).first()
        if agent.author_id != self.request.user.id:
            raise SuspiciousOperation('Only agent authors may add contributors')
        serializer.save()


class AgentLikesViewSet(mixins.CreateModelMixin,
                        mixins.RetrieveModelMixin,
                        mixins.DestroyModelMixin,
                        mixins.ListModelMixin,
                        viewsets.GenericViewSet):
    queryset = AgentLike.objects.all()
    serializer_class = AgentLikeSerializer
    pagination_class = StandardResultsSetPagination

    def get_queryset(self):
        qs = Q(agent__public=True)
        if self.request.user.is_authenticated:
            qs |= Q(user=self.request.user)
            qs |= Q(agent__author=self.request.user)
            qs |= Q(agent__contributors__user=self.request.user)
        elif self.action not in ('retrieve', 'list'):
            raise SuspiciousOperation('Unauthenticated users may not edit likes')

        if 'agent_id' in self.request.query_params:
            qs &= Q(agent_id=self.request.query_params['agent_id'])

        return self.queryset.filter(qs).distinct().order_by('id')

    def list(self, request, *args, **kwargs):
        if 'count' in self.request.query_params:
            return Response({'count': self.get_queryset().count()})
        if 'liked' in self.request.query_params:
            liked = self.get_queryset().filter(user=self.request.user).first()
            return Response({
                'liked': liked is not None,
                'like_id': -1 if liked is None else liked.id,
            })
        super(AgentLikesViewSet, self).retrieve(request, *args, **kwargs)

    def perform_create(self, serializer):
        if not self.request.user.is_authenticated:
            raise SuspiciousOperation('Unauthenticated users may not add likes')
        try:
            serializer.save()
        except IntegrityError:
            raise SuspiciousOperation('You have already liked this agent')


class PaymentIntentViewSet(mixins.CreateModelMixin, viewsets.GenericViewSet):
    queryset = PaymentIntent.objects.all()
    serializer_class = PaymentIntentSerializer

    def create(self, request, *args, **kwargs):
        intent = stripe.PaymentIntent.create(
            amount=30,  # £0.30
            currency='gbp',
            metadata={'integration_check': 'accept_a_payment'},
        )
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save(user=self.request.user, client_secret=intent['client_secret'])
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


class PaymentViewSet(mixins.CreateModelMixin, viewsets.GenericViewSet):
    queryset = Payment.objects.all()
    serializer_class = PaymentSerializer

    def create(self, request, *args, **kwargs):
        payload = request.body
        sig_header = request.META['HTTP_STRIPE_SIGNATURE']

        try:
            event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        except ValueError:
            # Invalid payload
            return HttpResponse(status=400)
        except stripe.error.SignatureVerificationError:
            # Invalid signature
            return HttpResponse(status=400)

        # Only supporting payment succeeded at the moment.
        if event.type != 'payment_intent.succeeded':
            return HttpResponse(status=400)

        client_secret = event.data.object['client_secret']
        payment_intent = PaymentIntent.objects.all().filter(client_secret=client_secret).get()
        user = payment_intent.user
        profile = user.profile

        if not profile:
            profile = UserProfile(user=user)

        profile.compute_credits += 10
        profile.save()

        payment = Payment(user=user, payment_reason=payment_intent.payment_reason)
        payment.save()

        return HttpResponse(status=200)


class ProfileViewSet(mixins.RetrieveModelMixin,
                     viewsets.GenericViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer

    def retrieve(self, request, *args, **kwargs):
        if self.kwargs['pk'] != 'me':
            return super(ProfileViewSet, self).retrieve(request, *args, **kwargs)
        instance = self.get_queryset().filter(user=self.request.user).first()
        if instance is None:
            instance = UserProfile(user=self.request.user)
            instance.save()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)


# start (agent id) -> runner instance id
# experience (runner instance id) -> ok
# save (runner instance id) -> ok (client can download after this call)
# stop (runner instance id) -> ok

class ComputeStartViewSet:
    pass
