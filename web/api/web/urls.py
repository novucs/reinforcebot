from django.conf.urls import url
from django.conf.urls.static import static
from django.urls import include, path
from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from rest_framework import permissions
from rest_framework_nested import routers

from web import settings, views

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
router.register(r'agents', views.AgentViewSet)
router.register(r'users', views.UserViewSet)
router.register(r'contributors', views.ContributorViewSet)
router.register(r'agent-likes', views.AgentLikesViewSet)
router.register(r'payment-intents', views.PaymentIntentViewSet)
router.register(r'payments', views.PaymentViewSet)
router.register(r'profiles', views.ProfileViewSet)
router.register(r'runners', views.RunnerViewSet)

runners_router = routers.NestedDefaultRouter(router, r'runners', lookup='runner')
runners_router.register(r'experience', views.RunnerExperienceViewSet)

urlpatterns = [
    url(r'^api/', include(router.urls)),
    url(r'^api/', include(runners_router.urls)),
    path('api/auth/', include('djoser.urls')),
    path('api/auth/', include('djoser.urls.jwt')),
    url(r'^api/swagger(?P<format>\.json|\.yaml)$', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    url(r'^api/swagger/$', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    url(r'^api/redoc/$', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    *static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT),
]
