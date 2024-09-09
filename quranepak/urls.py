from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.quran_api_view, name='quran_api'),
]
