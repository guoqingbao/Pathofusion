# Author: Guoqing Bao
# School of Computer Science, The University of Sydney
# Date: 2019-12-12
# GitHub Project Link: https://github.com/guoqingbao/Pathofusion
# Please cite our work if you found it is useful for your research or clinical practice

from django.shortcuts import render
from django.http import HttpResponse
from .models import imagelist, menu_selection, ihclist
from django.shortcuts import redirect
from django.conf import settings

# Create your views here.
def result(request):
    if not request.user.is_authenticated:
        return redirect('/login')
    user_config_menu = None
    try:
        user_config_menu = menu_selection.objects.get(user=request.user.username)
    except Exception as identifier:
        pass

    imgs = imagelist.objects.all()

    ihcs = ihclist.objects.all()
    ihc_types = list(set([item.type for item in ihcs]))
    if 'menu' in request.GET:
        if request.GET['menu'] in ihc_types:
            menu = request.GET['menu']
            user_config_menu.menu = menu
            imgs = ihclist.objects.all()
            print("user selected: ", user_config_menu.menu)


    unlabled_num = 0
    labled_num = 0
    for i in range(imgs.count()):
        if imgs[i].labels == "null" or imgs[i].labels == "[]":
            unlabled_num = unlabled_num + 1;
        else:
            labled_num = labled_num +1



    context = {'images':imgs, 'ihcs':ihc_types, 'label_index':0, 'unlabel_index':0,'haverow':0, 'haverow1':0, 'unlabled_num':unlabled_num, 'labled_num':labled_num, 'menu':user_config_menu}
    # print(imgs[0].thumb_image)
    return render( request,'imagelist/index.html', context)


