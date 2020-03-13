from djoser.serializers import UserCreateSerializer, UserSerializer
from rest_framework import serializers
from simple_history.utils import update_change_reason

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
        fields = ('id', 'name', 'description', 'parameters', 'author', 'changeReason')
        read_only_fields = ('author',)

    changeReason = serializers.CharField(write_only=True)

    def create(self, validated_data):
        change_reason = validated_data.pop('changeReason', None)
        instance = super(AgentSerializer, self).create(validated_data)
        update_change_reason(instance, change_reason)
        return instance

    def update(self, instance, validated_data):
        change_reason = validated_data.pop('changeReason', None)
        instance = super(AgentSerializer, self).update(instance, validated_data)
        update_change_reason(instance, change_reason)
        return instance


class AgentRetrieveSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Agent
        fields = ('id', 'name', 'description', 'parameters', 'author', 'history')
        read_only_fields = ('author', 'history',)

    history = HistoricalRecordField(read_only=True)
