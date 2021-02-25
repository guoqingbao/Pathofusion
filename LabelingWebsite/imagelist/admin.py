from django.contrib import admin

# Register your models here.
from .models import imagelist, menu_selection, ihclist
admin.site.register(imagelist)
admin.site.register(menu_selection)
admin.site.register(ihclist)