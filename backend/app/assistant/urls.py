from django.urls import path
from . import views

urlpatterns = [
    path('lyrics/', views.LyricsView.as_view(), name='lyrics'),
]
