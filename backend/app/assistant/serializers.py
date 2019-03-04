from rest_framework import serializers
from app.assistant.models import Lyric
from collections import OrderedDict


class LyricSerializer(serializers.ModelSerializer):
    class Meta:
        model = Lyric
        fields = '__all__'

    # def to_representation(self, instance):
    #     result = OrderedDict()
    #     result[str(instance.id)] = {"title": instance.title, "content": instance.content}
    #     return result