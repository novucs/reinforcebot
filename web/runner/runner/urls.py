from django.urls import path

from runner import views
from runner.settings import BASE_PATH

urlpatterns = [
    path(f'{BASE_PATH}/session/', views.handle_sessions),
    path(f'{BASE_PATH}/session/<int:session_id>/', views.handle_session),
    path(f'{BASE_PATH}/session/<int:session_id>/experience/', views.handle_session_experience),
]
