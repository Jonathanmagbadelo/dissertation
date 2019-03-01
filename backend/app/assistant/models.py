from django.db import models
import uuid


class Lyric(models.Model):
    RAP = 'RAP'
    ROCK = 'ROCK'
    POP = 'POP'

    CLASSIFICATIONS = (
        (RAP, 'Rap'),
        (ROCK, 'Rock'),
        (POP, 'Pop'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=30)
    content = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    classification = models.CharField(max_length=5, choices=CLASSIFICATIONS, default=POP)
