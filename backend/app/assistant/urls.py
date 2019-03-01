from django.urls import path
from . import views

urlpatterns = [
    path('api/lyric/', views.LyricListCreate.as_view()),
]
