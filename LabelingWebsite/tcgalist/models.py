from django.db import models

# Create your models here.
class tcgapatient(models.Model):
    id = models.AutoField(primary_key=True)
    tid = models.TextField()
    gender = models.TextField()
    birth = models.IntegerField()
    race = models.TextField()
    os = models.IntegerField()
    num = models.IntegerField()
    path = models.TextField()
    def __str__(self):
        return str(self.tid)
    class Meta:
        verbose_name_plural = "tcgapatient"