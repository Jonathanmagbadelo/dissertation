from django.urls import path

from . import views

urlpatterns = [
    path('lyrics/', views.LyricsListView.as_view(), name='lyrics'),
    path('lyrics/new/', views.CreateLyricView.as_view(), name='create-lyric'),
    path('lyrics/<uuid:pk>/', views.LyricsDetailView.as_view(), name='lyrics-detail'),
    path('songifai/suggest/', views.SuggestWordView.as_view(), name='suggested-words'),
    path('songifai/predict/', views.PredictWordView.as_view(), name='predicted-words')
]
