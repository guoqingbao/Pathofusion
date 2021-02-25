from django.conf.urls import url
from . import views
from django.urls import path, include

urlpatterns = [
    # url(r'labelling/submit', views.save, name='save'),
    # url(r'^$', views.index, name='index'),
    # url(r'^submit', views.save, name='save'),
    # url(r'^$', views.index, name='index')
    path('labelling/<int:id>/', views.index, name='index'),
    path('labelling/<int:id>', views.index, name='index'),

    path('labelling/submit', views.save, name='save')

]