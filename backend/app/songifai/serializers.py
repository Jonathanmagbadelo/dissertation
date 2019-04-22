from rest_framework import serializers
from app.songifai.models import Lyric


class LyricSerializer(serializers.ModelSerializer):
    class Meta:
        model = Lyric
        fields = '__all__'
