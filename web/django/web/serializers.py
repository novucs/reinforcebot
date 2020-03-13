from djoser.serializers import UserCreateSerializer, UserSerializer
from rest_framework import serializers

from web.models import Agent

additional_user_fields = ('first_name', 'last_name')
UserSerializer.Meta.fields += additional_user_fields
UserCreateSerializer.Meta.fields += additional_user_fields


class HistoricalRecordField(serializers.ListField):
    child = serializers.DictField()

    def to_representation(self, data):
        return super().to_representation(data.values())


class AgentSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Agent
        fields = ('id', 'name', 'description', 'parameters', 'author')
        read_only_fields = ('author',)


class AgentRetrieveSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Agent
        fields = ('id', 'name', 'description', 'parameters', 'author', 'history')
        read_only_fields = ('author', 'history',)

    history = HistoricalRecordField(read_only=True)
