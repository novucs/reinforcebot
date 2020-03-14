from django.db import models
from django.contrib.auth import get_user_model
from simple_history.models import HistoricalRecords


class Agent(models.Model):
    name = models.TextField()
    description = models.TextField()
    public = models.BooleanField(default=False)
    parameters = models.FileField(upload_to='agent-parameters/%Y/%m/%d/%H%M%S%f')
    author = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    history = HistoricalRecords()


class Contributor(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='contributors')


class Payment(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.SET_NULL, null=True)
