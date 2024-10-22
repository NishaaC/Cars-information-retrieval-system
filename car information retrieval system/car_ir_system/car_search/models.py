from django.db import models

class Car(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField(blank=True)
    imdb_rating = models.FloatField(null=True, blank=True)
    document = models.FileField(upload_to='car_documents')

    def __str__(self):
        return self.title