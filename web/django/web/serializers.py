from djoser.serializers import UserCreateSerializer, UserSerializer
from rest_framework import serializers

from web.models import Agent

additional_user_fields = ('first_name', 'last_name')
UserSerializer.Meta.fields += additional_user_fields
UserCreateSerializer.Meta.fields += additional_user_fields


class AgentSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Agent
        fields = ('name', 'author')
        read_only_fields = ('author',)
