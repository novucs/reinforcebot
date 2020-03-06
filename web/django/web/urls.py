from django.conf.urls import url
from django.urls import include, path
from rest_framework import routers

from web.api import AgentViewSet

router = routers.DefaultRouter()
router.register(r'agents', AgentViewSet)

urlpatterns = [
    url(r'^', include(router.urls)),
    path('auth/', include('djoser.urls')),
    path('auth/', include('djoser.urls.jwt')),
]
