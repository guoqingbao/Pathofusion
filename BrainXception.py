# Author: Guoqing Bao
# School of Computer Science, The University of Sydney
# Date: 2019-12-12
# GitHub Project Link: https://github.com/guoqingbao/Pathofusion
# Please cite our work if you found it is useful for your research or clinical practice

#%%
# from IPython import get_ipython

#%%
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
# from sklearn import cross_validation, metrics
from sklearn.metrics import f1_score,confusion_matrix, classification_report, accuracy_score
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K

import os
import gc
gc.enable()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.40
config.gpu_options.allow_growth=True
from io import StringIO

# import helper function (database manupulation, image augmentation, plot performance, train, etc.)
from models.helper import *



#%%


#%% [markdown]
# ## Load the pathology image datasets (two resolutions)

#%%
project_path = './'


#%%
path = project_path +'results/xception/' #the analysis results using xception going here
conn_256 = create_or_open_db(project_path + "data/brain_labeling_256.db")
conn_512 = create_or_open_db(project_path + "data/brain_labeling_512.db")


#%%
df = get_image_ids(conn_256, -1)


#%%
len(df)


#%%
# xception only accept one branch input
import keras
class DataGenerator(keras.utils.Sequence):

    def __init__(self, data, labels, augment = False, batch_size=32, classes=6):
        self.batch_size = batch_size
        self.labels = labels
        self.data = data
        self.aug = augment
        self.indexes = np.arange(len(self.data))
        self.classes = classes

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, index):
        start = index*self.batch_size
        end = (index+1)*self.batch_size
        frame_256 =  get_images(conn_256, list(self.data[start:end]))
        X1 = preprocess_images(frame_256,self.aug, 256)
        del frame_256
        y = to_categorical(self.labels[start:end], num_classes=self.classes)
        return X1, y


#%%


#%% [markdown]
# ## Training validation data and test data

#%%
train_val_data = df[~df.PATIENT_ID.isin(test_patient_ids)]
train_val_data = train_val_data.sample(frac=1, random_state=9) #shuffle the training/cross-validation data
train_val_data.reset_index(drop=True,inplace=True)
independentTest = df[df.PATIENT_ID.isin(test_patient_ids)]
independentTest = independentTest.sample(frac=1, random_state=11) #shuffle the test data

#%% [markdown]
# # We measure the performance of Xception (pretrained on Imagenet)

#%%
# train on full training/validation data and evaluate on test data, since we only care about the last epoch performance
x_train, y_train, x_test, y_test = train_val_data['ID'], train_val_data['TYPE']-1, independentTest['ID'], independentTest['TYPE']-1


#%%
from keras.applications import Xception
from keras.callbacks import LearningRateScheduler
import math
def cosine_lr(base_lr, e, epochs):
    lr = 0.5 * base_lr * (math.cos(math.pi * e / epochs) + 1)
    return lr

def step_decay(epoch):
    lr = cosine_lr(0.005, epoch, 18)
    return max(lr, 0.00005)

lrate = LearningRateScheduler(step_decay, verbose=1)

base_model = Xception(include_top=False, weights='imagenet')

x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=SGD(lr=0.005, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# the training epoch was decided based on 1/3 epoch of scratch-model 
train_history = model.fit_generator(DataGenerator(x_train, y_train, augment=True, batch_size=32), use_multiprocessing=False, workers = 2,
                                validation_data=DataGenerator(x_test, y_test, augment=False, batch_size=32), 
                                validation_steps = int(len(x_test)/ 32 + 1), callbacks=[lrate],
                    steps_per_epoch=int(len(x_train)/ 32) + 1, epochs=18)

model.save_weights(path + 'transfer_model.h5') #model saved over here after training
#evaluate on test data
probas_ = model.predict_generator(DataGenerator(x_test, y_test,augment=False, batch_size=32),steps = int(len(x_test)/32) + 1)
probas_ = probas_[:len(x_test)]
pred = np.argmax(probas_,axis=1) 
ac = accuracy_score(y_test, pred)
print("Testing accuracy with transfer {}\r\n".format(ac))


#%%
# additional performance metrics and train/test history
trains, tests = show_train_history([train_history], 'acc', 'val_acc', path, 'trans_train_test_history')
precision_recall_fscore = []

prf = precision_recall_fscore_support(y_test, pred,average = "weighted")
ac = accuracy_score(y_test, pred)

precision_recall_fscore.append([prf[0],prf[1],prf[2],ac])

metrics = pd.DataFrame(np.array(precision_recall_fscore), columns=['precision','recall','f1-score','accuracy'])
mean_values = []
for i in range(4):
    mean_values.append(np.mean(np.array(precision_recall_fscore)[:,i]))
metrics = metrics.append(pd.Series(mean_values, index=metrics.columns, name="Average"))
metrics.to_excel(path + 'test_metrics_transfer.xlsx')
metrics


#%%
# and its ROC performance
yts=[]
yts.append(y_test)
pbs=[]
pbs.append(probas_)
mean_tpr, auc_values = roc_plot(6,yts,pbs, path, 'roc_6class_test_transfer')


#%%
# we save the ROC performance, we compare it with others later
np.save(path + 'roc_6class_trans_mean_tpr.npy',mean_tpr)
np.save(path + 'roc_6class_trans_auc_values.npy',auc_values)


#%%


#%% [markdown]
# # We measure the performance of Xception (without pretraining on ImageNet), training from scratch

#%%
def step_decay(epoch):
    lr = cosine_lr(0.005, epoch, 50)
    return max(lr, 0.00005)
lrate = LearningRateScheduler(step_decay, verbose=1)

base_model = Xception(include_top=False, weights=None)

x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(6, activation='softmax')(x)

modelNotrans = Model(inputs=base_model.input, outputs=predictions)

modelNotrans.compile(optimizer=SGD(lr=0.005, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# training from scrtach takes longer epochs
train_history_notrans = modelNotrans.fit_generator(DataGenerator(x_train, y_train, augment=True, batch_size=32), use_multiprocessing=False, workers = 3,
                                validation_data=DataGenerator(x_test, y_test, augment=False, batch_size=32), 
                                validation_steps = int(len(x_test)/ 32 + 1), callbacks=[lrate],
                    steps_per_epoch=int(len(x_train)/ 32) + 1, epochs=50)
modelNotrans.save_weights(path + 'no_transfer_model.h5')
probas_notrans_ = modelNotrans.predict_generator(DataGenerator(x_test, y_test,augment=False, batch_size=32),steps = int(len(x_test)/32) + 1)
probas_notrans_ = probas_notrans_[:len(x_test)]
pred_notrans = np.argmax(probas_notrans_,axis=1) 
ac = accuracy_score(y_test, pred_notrans)
print("Testing accuracy with transfer {}\r\n".format(ac))


#%%
# additional performance metrics and train/test history, became stable after 20 epochs
trains, tests = show_train_history([train_history_notrans],'acc','val_acc',path,'notrans_train_test_history')
# np.save(path + "notrans_train_history.npy",np.array(trains))
# np.save(path + "notrans_test_history.npy",np.array(tests))
precision_recall_fscore = []

prf = precision_recall_fscore_support(y_test, pred_notrans,average = "weighted")
ac = accuracy_score(y_test, pred_notrans)

precision_recall_fscore.append([prf[0],prf[1],prf[2],ac])

metrics = pd.DataFrame(np.array(precision_recall_fscore), columns=['precision','recall','f1-score','accuracy'])
mean_values = []
for i in range(4):
    mean_values.append(np.mean(np.array(precision_recall_fscore)[:,i]))
metrics = metrics.append(pd.Series(mean_values, index=metrics.columns, name="Average"))
metrics.to_excel(path + 'test_metrics_notrans.xlsx')

#performance lower than model pretrained on ImageNet
metrics


#%%
# its ROC performance, still lower than model pretrained on ImageNet
yts=[]
yts.append(y_test)
pbs=[]
pbs.append(probas_notrans_)
mean_tpr,auc_values = roc_plot(6,yts,pbs, path, 'roc_6class_test_notransfer')


#%%
#save for later comparison
np.save(path + 'roc_6class_notrans_mean_tpr.npy',mean_tpr)
np.save(path + 'roc_6class_notrans_auc_values.npy',auc_values)


#%%



