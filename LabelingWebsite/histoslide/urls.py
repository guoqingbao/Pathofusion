from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^(?P<slide_id>\w+)/$', views.slide),
    url(r'^(?P<slug>\w+).dzi$', views.dzi),
    url(r'^(?P<slug>\w+).dzi.json$', views.properties),
    url(r'^(?P<slug>\w+)_files/(?P<level>\d+)/(?P<col>\d+)_(?P<row>\d+)\.(?P<slideformat>jpeg|png)$', views.dztile),
    url(r'^(?P<slug>\w+)_map/(?P<level>\d+)/(?P<col>\d+)_(?P<row>\d+)\.(?P<slideformat>jpeg|png)$', views.gmtile),
]
