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
    path(
        'chat/expression-table/<uuid:request_id>/',
        views.api_expression_table,
        name='api_expression_table',
    ),
    path(
        'chat/expression-table/<uuid:request_id>/download/',
        views.api_expression_table_download,
        name='api_expression_table_download',
    ),
]
