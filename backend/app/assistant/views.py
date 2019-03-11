# Create your views here.
from app.assistant.models import Lyric
from app.assistant.serializers import LyricSerializer
from rest_framework import generics
from django.views import View
from django.http import HttpResponse
from app.assistant.utils import utils


class LyricsView(generics.ListCreateAPIView):
    serializer_class = LyricSerializer

    def get_queryset(self):
        queryset = Lyric.objects.all()
        lyric_uuid = self.request.query_params.get('id', None)
        if lyric_uuid is not None:
            queryset = queryset.filter(id=lyric_uuid)
        return queryset


class NewLyricView(generics.CreateAPIView):
    authentication_classes = []
    serializer_class = LyricSerializer


class SuggestWordView(View):

    def get(self, request, *args, **kwargs):
        return HttpResponse("Hello")
