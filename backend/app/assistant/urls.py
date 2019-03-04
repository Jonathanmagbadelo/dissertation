from django.urls import path
from . import views

urlpatterns = [
    path('api/lyric/', views.LyricsView.as_view(), name='lyrics'),
]
