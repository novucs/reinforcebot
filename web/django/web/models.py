from django.db import models
from django.contrib.auth import get_user_model
from simple_history.models import HistoricalRecords


class Agent(models.Model):
    name = models.TextField()
    # parameters = models.FileField()
    history = HistoricalRecords()


class Payment(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.SET_NULL, null=True)
