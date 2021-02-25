__author__ = 'Bob'
from django import template
from django.contrib.auth.models import Group

register = template.Library()

@register.filter(name='hasGroup')
def hasGroup(user, group_name):
    try:
        group = Group.objects.get(name=group_name)
        return True if group in user.groups.all() else False
    except Exception as identifier:
        pass
    
    return False