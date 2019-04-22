from app.assistant.models import Lyric
from app.assistant.serializers import LyricSerializer
from rest_framework import generics
from django.views import View
from django.http import JsonResponse
from app.assistant.utils import utils


class LyricsListView(generics.ListCreateAPIView):
    serializer_class = LyricSerializer

    def get_queryset(self):
        queryset = Lyric.objects.all()
        lyric_uuid = self.request.query_params.get('id', None)
        if lyric_uuid is not None:
            queryset = queryset.filter(id=lyric_uuid)
        return queryset


class LyricsDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = LyricSerializer
    queryset = Lyric.objects.all()

    def get(self, request, *args, **kwargs):
        lyric = Lyric.objects.get(id=kwargs['pk'])
        serializer = self.get_serializer(lyric)
        return JsonResponse(data=serializer.data)


class CreateLyricView(generics.CreateAPIView):
    serializer_class = LyricSerializer


class SuggestWordView(View):
    def get(self, request, *args, **kwargs):
        return JsonResponse({'data': utils.suggest_words("king")})


class PredictWordView(View):
    def get(self, request, *args, **kwargs):
        return JsonResponse({'data': utils.suggest_words("TODO")})
