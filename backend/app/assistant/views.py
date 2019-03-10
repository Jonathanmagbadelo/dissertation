# Create your views here.
from app.assistant.models import Lyric
from app.assistant.serializers import LyricSerializer
from rest_framework import generics


class LyricsView(generics.ListCreateAPIView):
    queryset = Lyric.objects.all()
    serializer_class = LyricSerializer
