from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('chat/', views.api_chat, name='api_chat'),
    path(
        'chat/status/<uuid:request_id>/',
        views.api_chat_status,
        name='api_chat_status',
    ),
]
