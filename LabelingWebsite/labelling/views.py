# Author: Guoqing Bao
# School of Computer Science, The University of Sydney
# Date: 2019-12-12
# GitHub Project Link: https://github.com/guoqingbao/Pathofusion
# Please cite our work if you found it is useful for your research or clinical practice

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from imagelist.models import imagelist, ihclist
from labelling.models import markers
from django.views.decorators.csrf import csrf_exempt
import json
from django.shortcuts import redirect
from django.conf import settings
from django.contrib.auth.models import Group
from django import template

# Create your views here.
def index(request, id):
    if not request.user.is_authenticated:
        return redirect('/login')

    strMenu = request.GET['menu']
    image = imagelist.objects.get(pid=id)

    if strMenu != "GBM": 
        image = ihclist.objects.get(type=strMenu, pid=id)

    marks = markers.objects.filter(menu=strMenu)

    # print(strMenu)
    context = {'pid':image.pid, 'menu':strMenu, 'image':image.image, 'labels':image.labels, 'marks':marks, 'size':request.GET['size']}
    # print(context)
    return render(request, 'labelling/index.html', context)

# @csrf_exempt
def save(request):
    if not request.user.is_authenticated:
        return redirect('/login')
    if request.method == 'POST':
        # print("processing save...")
        if 'pid' in request.POST:
            
            points = request.POST.get('points')
            if points:
                points = json.loads(points)
            # print(request.POST)
            pid = request.POST['pid']
            menu = request.POST['curMenu']
            ihcs = ihclist.objects.all()
            ihc_types = list(set([item.type for item in ihcs]))
            # print(menu)
            if menu != "GBM" and menu in ihc_types:
                image = ihclist.objects.get(type=menu, pid=pid)
            else:
                image = imagelist.objects.get(pid=pid)
            
            if image != None:
                image.labels = points
                image.save()
                # print("saved")

                return HttpResponse("success")
                
    # nothing went well
    return HttpRepsonse('FAIL!!!!!')