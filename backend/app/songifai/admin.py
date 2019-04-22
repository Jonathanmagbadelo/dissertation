from django.contrib import admin
from .models import Lyric  # add this


class LyricAdmin(admin.ModelAdmin):  # add this
    list_display = ('title', 'content', 'classification')  # add this


# Register your models here.
admin.site.register(Lyric, LyricAdmin)  # add this
