from django.shortcuts import render

# Create your views here.
from app.assistant.models import Lyric
from app.assistant.serializers import LyricSerializer
from rest_framework import generics, viewsets


class LyricView(viewsets.ModelViewSet):
    queryset = Lyric.objects.all()
    serializer_class = LyricSerializer
