from django.db import models

class Car(models.Model):
    year = models.IntegerField()
    mileage = models.FloatField()
    tax = models.FloatField()
    mpg = models.FloatField()
    engine_size = models.FloatField()
    model = models.CharField(max_length=50)
    transmission = models.CharField(max_length=20)
    fuel_type = models.CharField(max_length=20)

    def __str__(self):
        return f"{self.year} {self.model} - {self.mileage} miles"

