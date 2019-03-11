from django.urls import path

from . import views

urlpatterns = [
    path('lyrics/', views.LyricsView.as_view(), name='lyrics'),
    path('lyrics/new', views.NewLyricView.as_view(), name='lyrics-new'),
    path('lyrics/delete/<uuid:id>/', views.DeleteLyricView.as_view(), name='lyrics-delete'),
]
