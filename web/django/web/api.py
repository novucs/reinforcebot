from django.contrib.auth import get_user_model
from rest_framework import mixins, viewsets
from rest_framework.pagination import PageNumberPagination
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated

from web.models import Agent
from web.serializers import AgentRetrieveSerializer, AgentSerializer, UserRetrieveSerializer


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
        return self.queryset.filter(author=self.request.user)

    def perform_create(self, serializer):
        serializer.save(author=serializer.context['request'].user)

    def get_serializer_class(self, *args, **kwargs):
        if self.action == 'retrieve':
            return AgentRetrieveSerializer
        return super(AgentViewSet, self).get_serializer_class()


class UserRetrieveViewSet(mixins.RetrieveModelMixin,
                          viewsets.GenericViewSet):
    queryset = get_user_model().objects.all()
    serializer_class = UserRetrieveSerializer
    permission_classes = (IsAuthenticated,)
