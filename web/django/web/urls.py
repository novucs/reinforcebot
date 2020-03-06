from django.conf.urls import url
from django.urls import include, path
from rest_framework import routers

from web.api import UserViewSet

router = routers.DefaultRouter()
# router.register(r'users', UserViewSet)

urlpatterns = [
    url(r'^', include(router.urls)),
    path('auth/', include('djoser.urls')),
    path('auth/', include('djoser.urls.jwt')),
]
