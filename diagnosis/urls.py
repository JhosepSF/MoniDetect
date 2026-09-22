"""
URLs configuration for diagnosis app.
"""
from django.urls import path
from . import views

app_name = 'diagnosis'

urlpatterns = [
    path('', views.index_view, name='index'),
    path('api/analyze/', views.analyze_view, name='analyze'),
    path('api/status/', views.model_status_view, name='status'),
]
