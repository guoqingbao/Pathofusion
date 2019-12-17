# Author: Guoqing Bao
# School of Computer Science, The University of Sydney
# Date: 2019-12-12
# GitHub Project Link: https://github.com/guoqingbao/Pathofusion
# Please cite our work if you found it is useful for your research or clinical practice

"""
The helper functions, such as database manipulation, image real-time augmentation, 
performance plotting, and training, are included in this module.
The database file should be pathology dataset (sqlite 3 format) we provided.
sqlite3, PIL, scikit-learn, keras and pytorch 1.2 (or above) should be installed before importing this module.
"""
import warnings;
warnings.filterwarnings('ignore');
import sqlite3
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
import gc
import random
import keras
import os
import math
import json
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
from PIL import ImageEnhance
from io import StringIO
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import torch
gc.enable()

test_patient_ids = [11, 23, 36]

def create_or_open_db(db_file):
    db_is_new = not os.path.exists(db_file)
    conn = sqlite3.connect(db_file, check_same_thread = False)
    if db_is_new:
        print('Creating schema')
        sql = '''create table if not exists PICTURES(
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        PATIENT_ID,
        TYPE INTEGER,
        GRADE INTEGER,
        ORIGIN_SIZE INTEGER,
        SIZE INTEGER,
        PICTURE BLOB     
        );'''
        conn.execute(sql) # shortcut for conn.cursor().execute(sql)
    else:
        print('Schema exists\n')
    
    return conn

def insert_picture(conn, patient_id,tp, grade, origin_sz, sz, image):
    sql = '''INSERT INTO PICTURES (PATIENT_ID, TYPE, GRADE, ORIGIN_SIZE, SIZE, PICTURE) VALUES(?, ?, ?, ?, ?, ?);'''
    conn.execute(sql,[patient_id,tp, grade, origin_sz, sz, sqlite3.Binary(image.tobytes())]) 
    conn.commit()

# we only focus on 6 types of morphological structures 
def get_patient_data(conn, patient_id):
    if patient_id == -1:
        sql = "SELECT *  FROM PICTURES WHERE TYPE != 7"
    else:
        sql = "SELECT * FROM PICTURES WHERE TYPE != 7 and PATIENT_ID = " + str(patient_id)

    return pd.read_sql_query(sql, conn)       

def get_patients(conn):

    sql = "SELECT ID, PATIENT_ID, TYPE  FROM PICTURES WHERE TYPE != 7"

    return pd.read_sql_query(sql, conn)  

def get_image_ids(conn, patient_id):
    if patient_id == -1:
        sql = "SELECT ID, TYPE, PATIENT_ID FROM PICTURES WHERE TYPE != 7"
    else:
        sql = "SELECT ID, TYPE, PATIENT_ID FROM PICTURES WHERE TYPE != 7 and PATIENT_ID = " + str(patient_id)

    return pd.read_sql_query(sql, conn)   

def get_images(conn, imageids):

    ids = ""
    for i in imageids:
        ids = ids + str(i) +","
    ids = ids[:-1]
    
    sql = "SELECT ID, TYPE, PICTURE FROM PICTURES WHERE TYPE != 7 and ID in (" + ids + ") ORDER BY " 
    
    sql = sql + "CASE ID "
    i = 1
    for id in imageids:
        sql = sql + " WHEN " + str(id) + " THEN " + str(i)
        i = i + 1

    sql += " END"

    d = pd.read_sql_query(sql, conn)
    return d 

def sqlite_insert_picture_quick(conn, patient_id, tp, grade, origin_sz, sz,x, y, image_binary):
    sql = "INSERT INTO PICTURES (PATIENT_ID, TYPE, GRADE, ORIGIN_SIZE, SIZE,X, Y, PICTURE) VALUES(?, ?, ?, ?, ?, ?, ?, ?)"
#     print(sql)
    conn.execute(sql,[patient_id, tp, grade, origin_sz, sz, x, y, image_binary]) 
    conn.commit()   

class History:
    def __init__(self):
        self.history = {'epoch':[], 'acc':[], 'val_acc':[], 'loss':[], 'val_loss':[]}
    def add(self, e, acc, val_acc, loss, val_loss):
        self.history['epoch'].append(e)
        self.history['acc'].append(acc)
        self.history['val_acc'].append(val_acc)        
        self.history['loss'].append(loss)        
        self.history['val_loss'].append(val_loss)  

def cosine_lr(opt, base_lr, e, epochs):
    lr = 0.5 * base_lr * (math.cos(math.pi * e / epochs) + 1)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return max(lr, 0.00005)


def accuracy(y, t):
    pred = y.data.max(1, keepdim=True)[1]
    acc = pred.eq(t.data.view_as(pred)).cpu().sum()
    return acc


class Logger:

    def __init__(self, log_dir, log_file, headers):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.f = open(os.path.join(log_dir, log_file), "w")
        header_str = "\t".join(headers + ["EndTime."])
        self.print_str = "\t".join(["{}"] + ["{:.6f}"] * (len(headers) - 1) + ["{}"])

        self.f.write(header_str + "\n")
        self.f.flush()
        print(header_str)

    def write(self, *args):
        now_time = datetime.now().strftime("%m/%d %H:%M:%S")
        self.f.write(self.print_str.format(*args, now_time) + "\n")
        self.f.flush()
        print(self.print_str.format(*args, now_time))

    def write_hp(self, hp):
        json.dump(hp, open(os.path.join(self.log_dir, "hp.json"), "w"))

    def close(self):
        self.f.close()

npimgs = []
def preprocess_images(frame, aug, sz):
    global npimgs
    for item in npimgs:
        del item
    
    npimgs = []  
    
    imgs_train = []
    for index, row in frame.iterrows():
        image = Image.frombytes(data=row.PICTURE,size = (sz,sz),mode="RGB")

        if aug == False:
            npimg = np.asarray(image)/255
        else:
            if random.randint(0,8) == 0:
                image = image.rotate(90)
            elif random.randint(0,8) == 5:
                image = image.rotate(-90)
            elif random.randint(0,8) == 7:
                image = image.rotate(180)
            
            if random.randint(0,8) == 0:
                enhancer = ImageEnhance.Sharpness(image)
                factor = 0.0
                npimg = np.asarray(enhancer.enhance(factor))/255
                del enhancer  
            elif random.randint(0,8) == 3:
                enhancer = ImageEnhance.Sharpness(image)
                factor = 2.0
                npimg = np.asarray(enhancer.enhance(factor))/255
                del enhancer  
            elif random.randint(0,8) == 7:
                enhancer = ImageEnhance.Contrast(image)
                factor = random.randint(4,9)/10
                npimg = np.asarray(enhancer.enhance(factor))/255
                del enhancer  
            else:
                npimg = np.asarray(image)/255

        if len(npimg.shape)==2:
            npimg = np.dstack((npimg,npimg,npimg))
        imgs_train.append(npimg)
        del image

    
    x = np.array(imgs_train)
    del imgs_train

    npimgs.append(x)   
    return x


class DataGenerator(keras.utils.Sequence):

    def __init__(self, data, labels, connections, image_sizes, augment = False, batch_size=32, classes=6):
        self.batch_size = batch_size
        self.labels = labels
        self.data = data
        self.aug = augment
        self.indexes = np.arange(len(self.data))
        self.classes = classes
        self.connections = connections
        self.image_sizes = image_sizes

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, index):
        start = index*self.batch_size
        end = (index+1)*self.batch_size
        xex = []
        for i in range(len(self.connections)):
            frame=  get_images(self.connections[i], list(self.data[start:end]))
            X1 = preprocess_images(frame,self.aug, self.image_sizes[i])
            X1 = np.transpose(X1, (0, 3,1,2))
            xex.append(X1)
            del frame
        y = self.labels[start:end]
        if len(xex) > 1:
            return xex, y
        else:
            return xex[0], y



def roc_plot(n_classes_, y_tests_,y_prediction_proba_, path, filename ):
    plt.rcParams['font.sans-serif']=['Arial']
    plt.rcParams['axes.unicode_minus']=False 
    plt.grid(linestyle = "--")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig = plt.gcf()
    fig.set_size_inches( 10, 8)

    tprs_all = []
    aucs_all = []
    mean_fpr = np.linspace(0, 1, 100)

    auc_values = []
    colors = ['aqua', 'black', 'cornflowerblue','green','yellow','darkorange','blue']
    for j in range(n_classes_):
        tprs = []
        aucs = []        
        for i in range(len(y_tests_)):
            fpr, tpr, thresholds = roc_curve(to_categorical(y_tests_[i],num_classes=n_classes_)[:, j], y_prediction_proba_[i][:, j])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs_all.append(interp(mean_fpr, fpr, tpr))

            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            aucs_all.append(roc_auc)


        if len(y_tests_)== 1:
            mean_tpr = tprs[0]
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            plt.plot(mean_fpr, mean_tpr, color=colors[j],
                     label=r'Class %d - ROC (AUC = %0.3f)' % (j+1, mean_auc),
                     lw=1.5, alpha=.6)
            auc_values.append(mean_auc)
            
        else:
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, color=colors[j],
                     label=r'Class %d - Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (j+1, mean_auc, std_auc),
                     lw=1.5, alpha=.6)
            auc_values.append(mean_auc)

    mean_tpr = np.mean(tprs_all, axis=0)
    mean_tpr[0] = .0
    mean_tpr[-1] = 1.0
    
    if n_classes_ > 1:
        plt.plot(mean_fpr, mean_tpr, color=colors[-1],
                 label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (np.mean(auc_values),  np.std(auc_values)),
                 lw=2, alpha=1)
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)



    std_tpr = np.std(tprs_all, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate',fontsize=12,fontweight='bold')
    plt.ylabel('True Positive Rate',fontsize=12,fontweight='bold')
    plt.legend(loc="lower right")


    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(path + filename + '.svg',format='svg')

    plt.show()
    return mean_tpr, auc_values

def roc_plot_compare(mean_tprs, auc_values, modelnames, colors, path, filename ):
    plt.rcParams['font.sans-serif']=['Arial']
    plt.rcParams['axes.unicode_minus']=False 
    plt.grid(linestyle = "--")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig = plt.gcf()
    fig.set_size_inches( 10, 8)
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(len(mean_tprs)):
        plt.plot(mean_fpr, mean_tprs[i], color=colors[i],
                     label=r'%s - ROC (AUC = %0.3f $\pm$ %0.3f)' % (modelnames[i], np.mean(auc_values[i]), np.std(auc_values[i])),
                     lw=2, alpha=1)
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate',fontsize=12,fontweight='bold')
    plt.ylabel('True Positive Rate',fontsize=12,fontweight='bold')
    plt.legend(loc="lower right")


    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(path + filename + '.svg',format='svg')

    plt.show()

def load_data(dataset_file, test_ids, ihc=False):
    conn = create_or_open_db(dataset_file)
    df = get_image_ids(conn, -1)
    conn.close()
    
    #for ihc image, convert type to positive (type>0) and negative expression (type==0)
    if ihc:
        df.loc[df.TYPE>0, 'TYPE'] = 1
    else:
        df.TYPE = df.TYPE - 1

    train_val_data = df[~df.PATIENT_ID.isin(test_ids)]
    train_val_data = train_val_data.sample(frac=1, random_state=9)
    train_val_data.reset_index(drop=True,inplace=True)
    x_train, y_train = train_val_data['ID'], train_val_data['TYPE']

    externalTest = df[df.PATIENT_ID.isin(test_ids)]
    externalTest = externalTest.sample(frac=1, random_state=11)
    externalTest.reset_index(drop=True,inplace=True)
    x_test, y_test = externalTest['ID'], externalTest['TYPE']
    return x_train, y_train, x_test, y_test


def train(model, trainLoader, testLoader, multiinputs, epochs, base_lr, weight_decay, log_path, log_file):
    model = nn.DataParallel(model).cuda()
    cudnn.benckmark = True

    opt = optim.SGD(model.parameters(),
                    lr=base_lr,
                    momentum=0.9,
                    weight_decay=weight_decay,
                    nesterov=True)
    loss_func = nn.CrossEntropyLoss().cuda()


    if testLoader != None:
        headers = ["Epoch", "LearningRate", "TrainLoss", "TestLoss", "TrainAcc.", "ValAcc." if log_file.find('_fold') >0 else "TestAcc."]
    else:
        headers = ["Epoch", "LearningRate", "TrainLoss", "TrainAcc."]

    logger = None  
    history = History()
    for e in range(epochs):
        lr = cosine_lr(opt, base_lr, e, epochs)
        model.train()
        train_loss, train_acc, train_n = 0, 0, 0
        bar = tqdm(total=len(trainLoader), leave=False)
        for i in range(len(trainLoader)):
            if multiinputs:
                [x1, x2], t = trainLoader[i]
                m1, m2, t = Variable(torch.FloatTensor(x1).cuda()), Variable(torch.FloatTensor(x2).cuda()), \
                                Variable(torch.LongTensor(t.tolist()).cuda())
                y = model(m1, m2)
            else:
                x1, t = trainLoader[i]
                m1, t = Variable(torch.FloatTensor(x1).cuda()), \
                                Variable(torch.LongTensor(t.tolist()).cuda())
                y = model(m1)
            loss = loss_func(y, t)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_acc += accuracy(y, t).item()
            train_loss += loss.item() * t.size(0)
            train_n += t.size(0)

            bar.set_description("Loss: {:.4f}, Accuracy: {:.2f}".format(
            train_loss / train_n, train_acc / train_n * 100), refresh=True)
            bar.update()
        bar.close()

        if testLoader!=None:
            if logger == None:
                logger = Logger(log_path, log_file, headers)  
            model.eval()
            val_loss, val_acc, val_n = 0, 0, 0
            probas_ = []

            with torch.no_grad():
                for i in range(len(testLoader)):
                    if multiinputs:
                        [x1, x2], t = testLoader[i]
                        m1, m2, t = Variable(torch.FloatTensor(x1).cuda()), Variable(torch.FloatTensor(x2).cuda()), \
                                        Variable(torch.LongTensor(t.tolist()).cuda())
                        y = model(m1, m2)
                    else:
                        x1, t = testLoader[i]
                        m1, t = Variable(torch.FloatTensor(x1).cuda()), \
                                        Variable(torch.LongTensor(t.tolist()).cuda())
                        y = model(m1)

                    if e == epochs-1:
                        probas_.extend(F.softmax(y).cpu().numpy().tolist())
                    loss = loss_func(y, t)
                    val_loss += loss.item() * t.size(0)
                    val_acc += accuracy(y, t).item()
                    val_n += t.size(0)
            logger.write(e+1, lr, train_loss / val_n, val_loss / val_n,
                        train_acc / train_n * 100, val_acc / val_n * 100) 
            history.add(e+1, train_acc / train_n * 100, val_acc / val_n * 100, train_loss / train_n, val_loss / val_n)            
        else:
            # logger.write(e+1, lr, train_loss / train_n, 
            #             train_acc / train_n * 100) 
            print("Epoch:", e+1, "\tLR:", np.round(lr,5), "\tTrain Loss:", np.round(train_loss / train_n,4), "\tTrain Acc:", np.round(train_acc / train_n * 100,4))
            history.add(e+1, train_acc / train_n * 100, -1, train_loss / train_n, -1)
    gc.collect()

    return history

def show_train_history(train_historys, train, validation, path, name, detailed=False):
    plt.rcParams['font.sans-serif']=['Arial']
    plt.rcParams['axes.unicode_minus']=False 
    plt.grid(linestyle = "--")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig = plt.gcf()
    fig.set_size_inches( 10, 8)
    trains = []
    validations = []
    
    for train_history in train_historys:
        trains.append(train_history[train])
        validations.append(train_history[validation])
    plt.plot(np.mean(trains,axis=0),lw=2)
    plt.plot(np.mean(validations,axis=0),lw=2)
    
    if detailed:
        for train_history in train_historys:
            plt.plot(train_history[train],lw=1, alpha=0.3)
            plt.plot(train_history[validation],lw=1, alpha=0.3)      

        
    plt.xlabel('Epoch',fontsize=12,fontweight='bold')
    plt.ylabel('Accuracy',fontsize=12,fontweight='bold')
#     plt.title('Train History Across 10 Fold',fontsize=12,fontweight='bold')
    plt.legend(['Train','Validation'],loc="lower right",fontsize=12)

    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)

    ax.set_ylim(30,100)
    plt.tight_layout()
    plt.savefig(path +name+'.svg',format='svg')

    
    plt.show()
    return trains, validations