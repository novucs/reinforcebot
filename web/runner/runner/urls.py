from django.conf.urls import url
from django.urls import include
from rest_framework import routers
from runner.settings import BASE_PATH

router = routers.DefaultRouter()
router.register(r'create', )

urlpatterns = [
    url(rf'^{BASE_PATH}/', include(router.urls)),
]
