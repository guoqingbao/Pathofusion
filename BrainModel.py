# Author: Guoqing Bao
# School of Computer Science, The University of Sydney
# Date: 2019-12-12
# GitHub Project Link: https://github.com/guoqingbao/Pathofusion
# Please cite our work if you found it is useful for your research or clinical practice

# from IPython import get_ipython
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import gc
gc.enable()
import warnings;
warnings.filterwarnings('ignore');
import tensorflow as tf
import sqlite3
# get_ipython().run_line_magic('matplotlib', 'inline')
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

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.40
config.gpu_options.allow_growth=True
from io import StringIO

# import helper function (database manupulation, image augmentation, plot performance, train, etc.)
from models.helper import *

# import our BCNN from models
from models.bcnn import BCNN


# # Load the pathology image datasets (two resolutions)
project_path = './'
path = project_path + "results/bcnn/"
conn_256 = create_or_open_db(project_path + "data/brain_labeling_256.db")
conn_512 = create_or_open_db(project_path + "data/brain_labeling_512.db")


df = get_image_ids(conn_256, -1)


len(df)

x_train, y_train, x_test, y_test = load_data(project_path + "data/brain_labeling_256.db", test_patient_ids, False) # patch ID and type
trainLoader = DataGenerator(x_train, y_train, connections=[conn_256, conn_512], image_sizes=[256,512], augment=True, classes=6)
testLoader = DataGenerator(x_test, y_test, connections=[conn_256, conn_512], image_sizes=[256,512], augment=False, classes=6)


# # Create BCNN model
model = BCNN(6)
print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
model = nn.DataParallel(model).cuda()


# # Load pretrained model if exists
if os.path.exists(path + 'torch_model.h5'):
    checkPoint = torch.load(path + 'torch_model.h5')
    model.load_state_dict(checkPoint)


# # Otherwise, train the model
if not os.path.exists(path + 'torch_model.h5'):
    history = train(model, trainLoader, None, multiinputs=True, epochs=50, base_lr=0.005, weight_decay=0.005, log_path=path, log_file='model_test.log')
    torch.save(model.state_dict(), path + 'torch_model.h5')




# # Evaluate on the test data
model.eval()
probas_ = []
for i in range(len(testLoader)):
    [x1, x2], t = testLoader[i]
    with torch.no_grad():
        m1, m2, t = Variable(torch.FloatTensor(x1).cuda()), Variable(torch.FloatTensor(x2).cuda()),                         Variable(torch.LongTensor(t.tolist()).cuda())
        y = model(m1, m2)
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
metrics.to_excel(path + 'test_metrics.xlsx')
metrics


#classes: 1 Necrosis-palisading, 2 MicVas-Proliferation, 3 Blood-Vessel, 4 Necrosis-Geo, 5 Brain-Tissue, 6 Tumor
# the test roc/auc
yts=[]
yts.append(y_test)
pbs=[]
pbs.append(probas_)
mean_tpr, auc_values = roc_plot(6,yts,pbs, path, 'roc_6class_test')


#save for later comparison
np.save(path + 'roc_6class_test_mean_tpr.npy',mean_tpr)
np.save(path + 'roc_6class_test_auc_values.npy',auc_values)


np.mean(auc_values)




# # Compare with other models

#xception transfer test
a1 = np.load(project_path + 'results/xception/roc_6class_trans_mean_tpr.npy')
b1 = np.load(project_path + 'results/xception/roc_6class_trans_auc_values.npy')

#xception no-transfer test
a2 = np.load(project_path + 'results/xception/roc_6class_notrans_mean_tpr.npy')
b2 = np.load(project_path + 'results/xception/roc_6class_notrans_auc_values.npy')

#subnet test
a3 = np.load(project_path + 'results/subnet/roc_6class_test_mean_tpr_subnet.npy')
b3 = np.load(project_path + 'results/subnet/roc_6class_test_auc_values_subnet.npy')

#bcnn test
a4 = np.load(project_path + 'results/bcnn/roc_6class_test_mean_tpr.npy')
b4 = np.load(project_path + 'results/bcnn/roc_6class_test_auc_values.npy')


roc_plot_compare([a1,a2,a3,a4], [b1,b2,b3,b4], ['Xception (Transfer Learning)','Xception','Subnet (256x256)','BCNN (256x256/512x512)'],  ['aqua','green','darkorange','blue'], path, 'auc_compared')




# # Compare with other patch resolutions
combinations = [[64,128],[64, 256], [64,512], [128,256], [128,512]]
tprs = []
aucs = []
names = []
other_resolution_path = project_path + 'results/bcnn_combination/'
for resolution in combinations:
    resolution_name = str(resolution[0]) + 'x' +str(resolution[0]) + '/' + str(resolution[1]) + 'x' + str(resolution[1])
    filename = str(resolution[0]) + '_' + str(resolution[1])
    tprs.append(np.load(other_resolution_path + 'roc_6class_test_mean_tpr_'+filename+'.npy'))
    aucs.append(np.load(other_resolution_path + 'roc_6class_test_auc_values_'+filename+'.npy'))
    names.append('BCNN '+resolution_name)
    
#the last one is image patches used in the paper
tprs.append(np.load(path + 'roc_6class_test_mean_tpr.npy'))
aucs.append(np.load(path + 'roc_6class_test_auc_values.npy'))
names.append('BCNN 256x256/512x512')    
roc_plot_compare(tprs, aucs, names,  ['aqua','green','darkorange','yellow', 'black', 'blue'],  path, 'auc_compared_resolution')




