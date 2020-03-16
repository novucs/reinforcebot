from django.db import models
from django.contrib.auth import get_user_model
from simple_history.models import HistoricalRecords


class Agent(models.Model):
    class Meta:
        unique_together = ('name', 'author_id')

    name = models.TextField()
    description = models.TextField()
    public = models.BooleanField(default=False)
    parameters = models.FileField(upload_to='agent-parameters/%Y/%m/%d/%H%M%S%f')
    author = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    history = HistoricalRecords()


class Contributor(models.Model):
    class Meta:
        unique_together = ('user_id', 'agent_id')

    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='contributors')


class AgentLike(models.Model):
    class Meta:
        unique_together = ('user_id', 'agent_id')

    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='likes')


DONATION = 'DONATION'
COMPUTE_CREDITS = 'COMPUTE_CREDITS'
PAYMENT_REASONS = (
    (COMPUTE_CREDITS, 'Compute Credits'),
)


class PaymentIntent(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.SET_NULL, null=True)
    client_secret = models.TextField()
    payment_reason = models.TextField(choices=PAYMENT_REASONS)
    history = HistoricalRecords()


class Payment(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.SET_NULL, null=True)
    payment_reason = models.TextField(choices=PAYMENT_REASONS)
    history = HistoricalRecords()


class UserProfile(models.Model):
    user = models.OneToOneField(get_user_model(), on_delete=models.CASCADE, related_name='profile')
    compute_credits = models.IntegerField(default=0)
