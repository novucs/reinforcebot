from django.contrib.auth import get_user_model
from rest_framework import (
    permissions,
    viewsets,
)
from rest_framework.generics import CreateAPIView

from web.serializers import (
    CreateUserSerializer,
    UserSerializer,
)


class UserViewSet(viewsets.ModelViewSet):
    queryset = get_user_model().objects.all()
    serializer_class = UserSerializer


class CreateUserView(CreateAPIView):
    model = get_user_model()
    permission_classes = [
        permissions.AllowAny  # Or anon users can't register
    ]
    serializer_class = CreateUserSerializer
