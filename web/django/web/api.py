from rest_framework import viewsets
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated

from web.models import Agent
from web.serializers import AgentSerializer, AgentRetrieveSerializer


class AgentViewSet(viewsets.ModelViewSet):
    queryset = Agent.objects.all()
    serializer_class = AgentSerializer
    permission_classes = (IsAuthenticated,)
    parser_classes = [MultiPartParser]

    def get_queryset(self):
        queryset = self.queryset
        query_set = queryset.filter(author=self.request.user)
        return query_set

    def perform_create(self, serializer):
        serializer.save(author=serializer.context['request'].user)

    def get_serializer_class(self, *args, **kwargs):
        if self.action == 'retrieve':
            return AgentRetrieveSerializer
        return super(AgentViewSet, self).get_serializer_class()
