from django.db import models

# Create your models here.
class SimulationResult(models.Model):
    run_date = models.DateTimeField(auto_now_add=True)
    num_rounds = models.IntegerField()
    num_devices = models.IntegerField()
    noise_multiplier = models.FloatField()
    accuracy = models.JSONField()
    loss = models.JSONField()
    leakage = models.JSONField()
    latency = models.JSONField()
    epsilon = models.JSONField()