# Author: Guoqing Bao
# School of Computer Science, The University of Sydney
# Date: 2019-12-12
# GitHub Project Link: https://github.com/guoqingbao/Pathofusion
# Please cite our work if you found it is useful for your research or clinical practice

# from IPython import get_ipython

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
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



project_path = './'
path = project_path + "results/bcnn/"

conn_256 = create_or_open_db(project_path + "data/cd276_labeling_256.db")
conn_512 = create_or_open_db(project_path + "data/cd276_labeling_512.db")



# for ihc images, "load_data" convert type to positive (type>0) and negative expression (type==0)
x_train, y_train, x_test, y_test = load_data(project_path + "data/cd276_labeling_256.db", test_ids=test_patient_ids, ihc=True) # patch ID and type
trainLoader = DataGenerator(x_train, y_train, connections=[conn_256, conn_512], image_sizes=[256,512], augment=True, classes=2)
testLoader = DataGenerator(x_test, y_test, connections=[conn_256, conn_512], image_sizes=[256,512], augment=False, classes=2)



model = BCNN(2, False)
print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))



# load the model if we have trained
if os.path.exists(path + 'torch_model_cd276.h5'):
    checkPoint = torch.load(path + 'torch_model_cd276.h5')
    model.load_state_dict(checkPoint)
    model = nn.DataParallel(model).cuda()


# otherwise, train the model and save (we only save the last epoch model)
if not os.path.exists(path + 'torch_model_cd276.h5'):
    history = train(model, trainLoader, testLoader, multiinputs=True, epochs=50, base_lr=0.0001, weight_decay=0.005, log_path=path, log_file='cd276_train_test.log')
    torch.save(model.state_dict(), path + 'torch_model_cd276.h5')
    np.save(path + 'train_test_history_cd276.npy', np.array(history.history))



history = np.load(path + 'train_test_history_cd276.npy',allow_pickle=True)



# let's plot the train/test history
trains, tests = show_train_history([history.item()], 'acc', 'val_acc', path, 'train_test_history_cd276')







#let's see the test results
model.eval()
probas_ = []
with torch.no_grad():
    for i in range(len(testLoader)):
        [x1, x2], t = testLoader[i]
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
metrics.to_excel(path + 'test_metrics_cd276.xlsx')
metrics



# the test roc/auc

yts=[]
yts.append(y_test)
pbs=[]
pbs.append(probas_)
mean_tpr, auc_values = roc_plot(2,yts,pbs, path, 'roc_2class_test_cd276')



#save for later comparison
np.save(path + 'roc_2class_test_mean_tpr_cd276.npy',mean_tpr)
np.save(path + 'roc_2class_test_auc_values_cd276.npy',auc_values)





