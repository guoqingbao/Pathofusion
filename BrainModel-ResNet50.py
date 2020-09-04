# Author: Guoqing Bao
# School of Computer Science, The University of Sydney
# Date: 2019-12-12
# GitHub Project Link: https://github.com/guoqingbao/Pathofusion
# Please cite our work if you found it is useful for your research or clinical practice

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import gc
gc.enable()
import warnings;
warnings.filterwarnings('ignore');
import sqlite3
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_fscore_support
# from sklearn import cross_validation, metrics
from sklearn.metrics import f1_score,confusion_matrix, classification_report, accuracy_score


import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from io import StringIO

# import helper function (database manupulation, image augmentation, plot performance, train, etc.)
from project.models.helper import *

# import our SubNet from models
from project.models.resnet50 import ResNet50

# # Load the pathology image datasets (one resolution)

project_path = './project/brain/'

path = project_path + "results/resnet50/"
conn_256 = create_or_open_db(project_path + "data/brain_labeling_256.db")


x_train, y_train, x_test, y_test = load_data(project_path + "data/brain_labeling_256.db", test_ids=test_patient_ids, ihc=False) # patch ID and type
trainLoader = DataGenerator(x_train, y_train, connections=[conn_256], image_sizes=[256], augment=True, classes=6)
testLoader = DataGenerator(x_test, y_test, connections=[conn_256], image_sizes=[256], augment=False, classes=6)


df = get_image_ids(conn_256, -1)

len(df)

# # Create a ResNet50 model

model = ResNet50(6)
print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

# # Load model if exists

# load the model if we have trained
if os.path.exists(path + 'torch_model_resnet50.h5'):
    checkPoint = torch.load(path + 'torch_model_resnet50.h5')
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(checkPoint)

# # Otherwise, train the model

if not os.path.exists(path + 'torch_model_resnet50.h5'):
    history = train(model, trainLoader, None, multiinputs=False, epochs=50, base_lr=0.005, weight_decay=0.005, log_path=path, log_file='resnet50_model.log')
    torch.save(model.state_dict(), path + 'torch_model_resnet50.h5')

# # Evalaute the model on test data

model.eval()
probas_ = []
for i in range(len(testLoader)):
    x1, t = testLoader[i]
    with torch.no_grad():
        m1, t = Variable(torch.FloatTensor(x1).cuda()),                         Variable(torch.LongTensor(t.tolist()).cuda())
        y = model(m1)
        probas_.extend(F.softmax(y).cpu().numpy().tolist())
probas_ = np.array(probas_)
pred = np.argmax(probas_,axis=1) 
ac = accuracy_score(y_test, pred)
print("External Testing accuracy {}\r\n".format(ac))


# and other test metrics
precision_recall_fscore = []

prf = precision_recall_fscore_support(y_test, pred,average = "weighted")
ac = accuracy_score(y_test, pred)

precision_recall_fscore.append([prf[0],prf[1],prf[2],ac])

metrics = pd.DataFrame(np.array(precision_recall_fscore), columns=['precision','recall','f1-score','accuracy'])
mean_values = []
for i in range(4):
    mean_values.append(np.mean(np.array(precision_recall_fscore)[:,i]))
metrics = metrics.append(pd.Series(mean_values, index=metrics.columns, name="Average"))
metrics.to_excel(path + 'test_metrics_resnet50.xlsx')
metrics


#classes: 1 Necrosis-palisading, 2 MicVas-Proliferation, 3 Blood-Vessel, 4 Necrosis-Geo, 5 Brain-Tissue, 6 Tumor
classesnames = ['Necrosis Palisading', 'Microvascular Proliferation', 'Blood Vessel', 'Necrosis Geographic', 'Brain Tissue', 'Tumor Background']
colors = ['red', 'blue', 'cyan','yellow','lightgrey','green','black']

# the test roc/auc
yts=[]
yts.append(y_test)
pbs=[]
pbs.append(probas_)
mean_tpr, auc_values = roc_plot(6,yts,pbs, classesnames, colors, path, 'roc_6class_test_resnet50')


#save for later comparison
np.save(path + 'roc_6class_test_mean_tpr_resnet50.npy',mean_tpr)
np.save(path + 'roc_6class_test_auc_values_resnet50.npy',auc_values)

np.mean(auc_values)




