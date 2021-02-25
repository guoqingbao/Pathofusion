# Author: Guoqing Bao
# School of Computer Science, The University of Sydney
# Date: 2019-12-12
# GitHub Project Link: https://github.com/guoqingbao/Pathofusion
# Please cite our work if you found it is useful for your research or clinical practice

from django.shortcuts import render
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.shortcuts import render
from .models import tcgapatient
from PIL import Image
import io
import os
import base64
from django.conf import settings
# Create your views here.


def listing(request):
    patient_list = tcgapatient.objects.all()
    paginator = Paginator(patient_list, 24) # Show 25 contacts per page

    page = request.GET.get('page')
    tcgapatients = paginator.get_page(page)

    menu = None
    try:
        menu = menu_selection.objects.get(user=request.user.username)
    except Exception as identifier:
        pass

    # data_url = 'data:image/jpg;base64,'
    for patient in tcgapatients:
        path, filename = os.path.split(patient.path + ".jpg")
        print(filename)
        patient.path = filename
        # img = Image.open(settings.TCGA_PATH + patient.path + ".jpg")
        # with io.BytesIO() as output:
        #     img.save(output, format="JPEG")
        #     contents = output.getvalue()
        #     patient.path = data_url + base64.b64encode(contents).decode()
        # img.close() 

    return render(request, 'index.html', {'tcgapatients': tcgapatients, 'menu':menu, 'label_index':0, 'haverow':0})