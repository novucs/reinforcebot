from django.conf.urls import url
from django.urls import include, path
from rest_framework import routers

from web.api import AgentViewSet

router = routers.DefaultRouter()
router.register(r'agents', AgentViewSet)

urlpatterns = [
    url(r'^api/', include(router.urls)),
    path('api/auth/', include('djoser.urls')),
    path('api/auth/', include('djoser.urls.jwt')),
]
