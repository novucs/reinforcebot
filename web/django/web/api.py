from django.contrib.auth import get_user_model
from django.contrib.postgres.search import TrigramDistance
from django.core.exceptions import SuspiciousOperation
from django.db.models import Q
from rest_framework import viewsets
from rest_framework.pagination import PageNumberPagination
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated

from web.models import Agent, Contributor
from web.serializers import AgentRetrieveSerializer, AgentSerializer, ContributorSerializer, UserRetrieveSerializer


class StandardResultsSetPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 1000


class AgentViewSet(viewsets.ModelViewSet):
    queryset = Agent.objects.all()
    serializer_class = AgentSerializer
    permission_classes = (IsAuthenticated,)
    parser_classes = [MultiPartParser, JSONParser]
    pagination_class = StandardResultsSetPagination

    def get_queryset(self):
        qs = Q(author=self.request.user)
        if self.action in ('partial_update', 'update'):
            qs |= Q(contributors__user=self.request.user)
        if self.action in ('retrieve', 'list'):
            qs |= Q(contributors__user=self.request.user)
            qs |= Q(public=True)
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
        return self.queryset.filter(qs).distinct()

    def get_serializer_class(self, *args, **kwargs):
        if self.action == 'retrieve':
            return AgentRetrieveSerializer
        return super(AgentViewSet, self).get_serializer_class()

    def perform_create(self, serializer):
        serializer.save(author=serializer.context['request'].user)

    def perform_update(self, serializer):
        if 'public' in self.request.data and serializer.instance.author_id != self.request.user.id:
            raise SuspiciousOperation('Only agent authors may change the publication status')
        serializer.save()


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = get_user_model().objects.all()
    serializer_class = UserRetrieveSerializer
    permission_classes = (IsAuthenticated,)
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


class ContributorViewSet(viewsets.ModelViewSet):
    queryset = Contributor.objects.all()
    serializer_class = ContributorSerializer
    permission_classes = (IsAuthenticated,)
    pagination_class = StandardResultsSetPagination

    def get_queryset(self):
        qs = Q(agent__author=self.request.user)
        if self.action == 'destroy':
            qs |= Q(user=self.request.user)
        if self.action in ('retrieve', 'list'):
            qs |= Q(user=self.request.user)
            qs |= Q(agent__contributors__user=self.request.user)
            qs |= Q(agent__public=True)
        return self.queryset.filter(qs).distinct()

    def perform_create(self, serializer):
        agent = Agent.objects.all().filter(id=self.request.data['agent_id']).first()
        if agent.author_id != self.request.user.id:
            raise SuspiciousOperation('Only agent authors may add contributors')
        serializer.save()
