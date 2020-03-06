from rest_framework import serializers

from web.models import Agent


class AgentSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Agent
        fields = ('name',)
