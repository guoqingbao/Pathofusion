# Author: Guoqing Bao
# School of Computer Science, The University of Sydney
# Date: 2019-12-12
# GitHub Project Link: https://github.com/guoqingbao/Pathofusion
# Please cite our work if you found it is useful for your research or clinical practice

from django.db import models

# Create your models here.
class imagelist(models.Model):
    pid = models.IntegerField(primary_key=True)
    thumb_image = models.TextField()
    image = models.TextField()
    labels = models.TextField()
    def __str__(self):
        return str(self.pid)
    class Meta:
        verbose_name_plural = "imagelist"

class menu_selection(models.Model):
    id = models.IntegerField(primary_key=True)
    user = models.TextField()
    menu = models.TextField()
    marker_size = models.IntegerField()
    localserver = models.BooleanField(default=False)
    
    def __str__(self):
        return str(self.user + "_" + self.menu)
    class Meta:
        verbose_name_plural = "menu_selection"

class ihclist(models.Model):
    pid = models.IntegerField()
    type = models.TextField()
    thumb_image = models.TextField()
    image = models.TextField()
    labels = models.TextField()
    def __str__(self):
        return str(self.pid) + "_" + self.type
    class Meta:
        verbose_name_plural = "ihclist"