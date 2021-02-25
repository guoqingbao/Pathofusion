from django.db import models

# Create your models here.
class markers(models.Model):
    id = models.IntegerField(primary_key=True)
    menu = models.TextField()
    class_id = models.IntegerField()
    name = models.TextField()
    color = models.TextField()
    def __str__(self):
        return str(self.menu + "_" + self.name)
    class Meta:
        verbose_name_plural = "markers"