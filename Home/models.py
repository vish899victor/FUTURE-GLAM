from django.db import models

# Create your models here.
class fields(models.Model):
    name = models.CharField(max_length= 100)
    img = models.ImageField(upload_to='pics')
    car = models.ImageField(upload_to='pics',default='images/DS.jpeg')
    desc = models.TextField()
    sal = models.IntegerField()