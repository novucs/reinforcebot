from django.conf.urls import url
from django.conf.urls.static import static
from django.urls import include, path
from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from rest_framework import permissions, routers

from web import settings
from web.api import AgentViewSet, UserRetrieveViewSet

schema_view = get_schema_view(
    openapi.Info(
        title="ReinforceBot API",
        default_version='v1',
        description="API for accessing ReinforceBot resources",
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

router = routers.DefaultRouter()
router.register(r'agents', AgentViewSet)
router.register(r'users', UserRetrieveViewSet)

urlpatterns = [
    url(r'^api/', include(router.urls)),
    path('api/auth/', include('djoser.urls')),
    path('api/auth/', include('djoser.urls.jwt')),
    url(r'^api/swagger(?P<format>\.json|\.yaml)$', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    url(r'^api/swagger/$', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    url(r'^api/redoc/$', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    *static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT),
]
