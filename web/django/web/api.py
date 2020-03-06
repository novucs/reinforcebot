from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated

from web.models import Agent
from web.serializers import AgentSerializer


class AgentViewSet(viewsets.ModelViewSet):
    queryset = Agent.objects.all()
    serializer_class = AgentSerializer
    permission_classes = (IsAuthenticated,)
