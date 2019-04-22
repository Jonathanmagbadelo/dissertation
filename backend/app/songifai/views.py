from app.songifai.models import Lyric
from app.songifai.serializers import LyricSerializer
from rest_framework import generics
from django.views import View
from django.http import JsonResponse
from app.songifai.utils import utils


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
        word = self.request.GET.get('word', '')
        return JsonResponse({'data': utils.suggest_words(word)})


class PredictWordView(View):
    def get(self, request, *args, **kwargs):
        return JsonResponse({'data': utils.predict_words(False, False, "TODO")})


class ChangeEmbeddingView(View):
    def get(self, request, *args, **kwargs):
        embedding = self.request.GET.get('embedding', '')
        return JsonResponse({'data': utils.load_embeddings(embedding)})


class ClassificationView(generics.RetrieveUpdateDestroyAPIView):

    def post(self, request, *args, **kwargs):
        lyrics = self.request.data['lyric']
        print(lyrics)
        return JsonResponse({'classification': utils.classify_lyrics(lyrics)})


