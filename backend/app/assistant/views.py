from django.shortcuts import render

# Create your views here.
from app.assistant.models import Lyric
from app.assistant.serializers import LyricSerializer
from rest_framework import generics, viewsets
from django.views import View


class LyricsView(View):
    queryset = Lyric.objects.all()[:1]
    queryset = {str(i.id): i for i in queryset}
    serializer_class = LyricSerializer
    def get(self):
        print("y")
