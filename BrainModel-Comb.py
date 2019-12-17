# Author: Guoqing Bao
# School of Computer Science, The University of Sydney
# Date: 2019-12-12
# GitHub Project Link: https://github.com/guoqingbao/Pathofusion
# Please cite our work if you found it is useful for your research or clinical practice

# from IPython import get_ipython

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import warnings;
warnings.filterwarnings('ignore');
import tensorflow as tf
import sqlite3
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from keras.preprocessing import image
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,confusion_matrix, classification_report, accuracy_score
import os
import gc
import math
gc.enable()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.40
config.gpu_options.allow_growth=True

# import helper function (database manupulation, image augmentation, plot performance, train, etc.)
from models.helper import *

# import our BDCNN from models
from models.bdcnn import BDCNN


# %%
project_path = './'


# %%
path = project_path + "results/bdcnn_combination/"
x_train, y_train, x_test, y_test = load_data(project_path + "data/brain_labeling_64.db", test_patient_ids, False) # we only need IDs and types at this stage

# %% [markdown]
# # Compare the performance of DRCNN using different patch resolutions

# %%
# the combination of image patches
tasks = [[64,128],[64, 256], [64,512], [128,256], [128,512]]

for resolution in tasks:
    # for each combination, we load corresponding image patches (a lower resolution + a higher resolution)
    resolution_name = str(resolution[0]) + '_' + str(resolution[1])
    print("\r\n**************** Task " + resolution_name + " ***************")
    conn_small= create_or_open_db(project_path + "data/brain_labeling_"+str(resolution[0])+".db")
    conn_large = create_or_open_db(project_path + "data/brain_labeling_"+str(resolution[1])+".db")

    trainLoader = DataGenerator(x_train, y_train, connections=[conn_small, conn_large], image_sizes=resolution, augment=True, classes=6)
    testLoader = DataGenerator(x_test, y_test, connections=[conn_small, conn_large], image_sizes=resolution, augment=False, classes=6)
    # construct our model
    model = BDCNN(6)
    print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))
    
    # train the model
    history = train(model, trainLoader, testLoader, multiinputs=True, epochs=50, base_lr=0.005, weight_decay=0.005, log_path=path, log_file='model_test_'+resolution_name+'.log')
    
    # save the training and test history
    np.save(path + 'train_test_history_'+resolution_name+'.npy', np.array(history))

    # test the model
    probas_ = []
    for i in range(len(testLoader)):
        [x1, x2], t = testLoader[i]
        with torch.no_grad():
            m1, m2, t = Variable(torch.FloatTensor(x1).cuda()), Variable(torch.FloatTensor(x2).cuda()),                             Variable(torch.LongTensor(t.tolist()).cuda())
            y = model(m1, m2)
            probas_.extend(F.softmax(y).cpu().numpy().tolist())
    probas_ = np.array(probas_)
    pred = np.argmax(probas_,axis=1) 
    ac = accuracy_score(y_test, pred)
    print("Test accuracy {}\r\n".format(ac))

    #additional test performance metrics
    precision_recall_fscore = []

    prf = precision_recall_fscore_support(y_test, pred,average = "weighted")
    ac = accuracy_score(y_test, pred)

    precision_recall_fscore.append([prf[0],prf[1],prf[2],ac])

    metrics = pd.DataFrame(np.array(precision_recall_fscore), columns=['precision','recall','f1-score','accuracy'])
    mean_values = []
    for i in range(4):
        mean_values.append(np.mean(np.array(precision_recall_fscore)[:,i]))
    metrics = metrics.append(pd.Series(mean_values, index=metrics.columns, name="Average"))
    metrics.to_excel(path + 'test_metrics_'+resolution_name+'.xlsx')
    print(metrics)

    # roc test performance
    yts=[]
    yts.append(y_test)
    pbs=[]
    pbs.append(probas_)
    mean_tpr, auc_values = roc_plot(6,yts,pbs, path, 'roc_6class_test_'+resolution_name)
    plt.show()
    np.save(path + 'roc_6class_test_mean_tpr_'+resolution_name+'.npy',mean_tpr)
    np.save(path + 'roc_6class_test_auc_values_'+resolution_name+'.npy',auc_values)
    print("*************************************\r\n")


# %%



