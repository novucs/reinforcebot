from django.contrib.auth import get_user_model
from djoser.serializers import UserCreateSerializer, UserSerializer
from rest_framework import serializers
from simple_history.utils import update_change_reason

from web.models import Agent, Contributor

additional_user_fields = ('first_name', 'last_name')
UserSerializer.Meta.fields += additional_user_fields
UserCreateSerializer.Meta.fields += additional_user_fields


class HistoricalRecordField(serializers.ListField):
    child = serializers.DictField()

    def to_representation(self, data):
        return super().to_representation(data.values())


class UserRetrieveSerializer(serializers.ModelSerializer):
    class Meta:
        model = get_user_model()
        fields = ('id', 'first_name', 'last_name', 'email', 'username')
        read_only_fields = fields


class ContributorViaAgentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Contributor
        fields = ('user',)


class AgentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Agent
        fields = ('id', 'name', 'description', 'public', 'parameters', 'author', 'changeReason', 'contributors')
        read_only_fields = ('author',)

    changeReason = serializers.CharField(write_only=True)
    contributors = ContributorViaAgentSerializer(many=True)

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


class AgentRetrieveSerializer(serializers.ModelSerializer):
    class Meta:
        model = Agent
        fields = ('id', 'name', 'description', 'public', 'parameters', 'author', 'history', 'contributors')
        read_only_fields = ('author', 'history',)

    history = HistoricalRecordField(read_only=True)
    contributors = ContributorViaAgentSerializer(many=True)


class ContributorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Contributor
        fields = ('id', 'user_id', 'agent_id', 'user', 'agent')

    agent_id = serializers.IntegerField(write_only=True)
    agent = AgentRetrieveSerializer(read_only=True)
    user_id = serializers.IntegerField(write_only=True)
    user = UserRetrieveSerializer(read_only=True)
