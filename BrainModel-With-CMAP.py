'''
Author: Guoqing Bao
School of Computer Science, The University of Sydney
Date: 2020-05-01
GitHub Project Link: https://github.com/guoqingbao/Pathofusion
Please cite our work if you found it is useful for your research or clinical practice

#Can be cited as:

Guoqing Bao, Manuel B. Graeber, Xiuying Wang, "A Bifocal Classification and Fusion Network for Multimodal Image Analysis in Histopathology", 
16th International Conference on Control, Automation, Robotics and Vision (ICARCV 2020), In Press.

'''

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import gc
gc.enable()
import warnings;
warnings.filterwarnings('ignore');
import tensorflow as tf
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

# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.40
# config.gpu_options.allow_growth=True
from io import StringIO

# import helper function (database manupulation, image augmentation, plot performance, train, etc.)
from project.models.helper import *

# import our BCNN from models
from project.models.bcnn import BCNN


# # Load the pathology image datasets (two resolutions)

project_path = './project/brain/'
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


#classes: 1 Necrosis-palisading, 2 MicVas-Proliferation, 3 Blood-Vessel, 4 Necrosis-Geo, 5 Brain-Tissue, 6 Tumor Background
classesnames = ['Necrosis Palisading', 'Microvascular Proliferation', 'Blood Vessel', 'Necrosis Geographic', 'Brain Tissue', 'Tumor Background']
colors = ['red', 'blue', 'cyan','yellow','lightgrey','green','black']

# the test roc/auc
yts=[]
yts.append(y_test)
pbs=[]
pbs.append(probas_)
mean_tpr, auc_values = roc_plot(6,yts,pbs, classesnames, colors, path, 'roc_6class_test')


#save for later comparison
np.save(path + 'roc_6class_test_mean_tpr.npy',mean_tpr)
np.save(path + 'roc_6class_test_auc_values.npy',auc_values)


np.mean(auc_values)

# # Compare with other models

def roc_plot_compare(mean_tprs, auc_values, modelnames, colors, path, filename ):
    # plt.rcParams['font.sans-serif']=['Arial']
    plt.rcParams['axes.unicode_minus']=False 
#     plt.grid(linestyle = "--")
    ax = plt.gca()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    fig = plt.gcf()
    fig.set_size_inches(5.5, 4.5)
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(len(mean_tprs)):
        plt.plot(mean_fpr, mean_tprs[i], color=colors[i],
                     label=r'%s - ROC (AUC=%0.3f$\pm$%0.3f)' % (modelnames[i], np.mean(auc_values[i]), np.std(auc_values[i])),
                     lw=1, alpha=1)
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',label='Chance')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    plt.xlabel('False Positive Rate',fontsize=10)
    plt.ylabel('True Positive Rate',fontsize=10)
    plt.legend(loc="lower right")


    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(path + filename + '.svg',format='svg')

    plt.show()


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


roc_plot_compare([a1,a2,a3,a4], [b1,b2,b3,b4], ['Xception (TL)','Xception','Subnet','BCNN'],  ['aqua','green','darkorange','black'], path, 'auc_compared')

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



# # For Conference
# ## 1. CAM Overlay to illustrate discriminative regions captured by BCNN

import random
indexes = []
tps = []
for j in range(1200):
    i = random.randint(0,len(testLoader))
    [x1, x2], t = testLoader[i]
    tp = int(t)
#     print(tp)
#     if tp in [0,1,3] and not tp in tps:
    if tp == 2 and len(tps)<3:
        tps.append(tp)
        indexes.append(i)
        if len(tps) ==3:
            break
print("Random choose test tiles ", indexes)
print(tps)


indexes =  [426, 1341, 311] # test tiles
tps = [3, 1, 2] # corresponding class ids


from PIL import Image

x1s = []
x2s = []
cam1s = []
cam2s = []
model.eval()
for k in indexes:
    [x1, x2], t = testLoader[k]
    x1s.append(x1)
    x2s.append(x2)
    gradients1 = None
    gradients2 = None
    features_blobs1 = []
    features_blobs2 = []

    def hook_feature1(module, input, result):
        features_blobs1.append(result.data.cpu().numpy())

    def hook_feature2(module, input, result):
        features_blobs2.append(result.data.cpu().numpy())

    def save_gradient1(module,  grad_in, grad_out):
        global gradients1
        if grad_out!=None:
            gradients1 = grad_out[0]

    def save_gradient2(module, grad_in, grad_out):
        global gradients2
        if grad_out!=None:
            gradients2 = grad_out[0]

    a = model.module.path1.block3[3][4].register_forward_hook(hook_feature1)
    b = model.module.path2.block3[3][4].register_forward_hook(hook_feature2)
    c = model.module.path1.block3[3][4].register_backward_hook(save_gradient1)
    d = model.module.path2.block3[3][4].register_backward_hook(save_gradient2)
    # with torch.no_grad():
    m1, m2 = Variable(torch.FloatTensor(x1).cuda()), Variable(torch.FloatTensor(x2).cuda())
    x = model(m1,m2)

    # model_output is the final output of the model (1, 1000)
    conv_output1, conv_output2 = features_blobs1[0], features_blobs2[0]
    target_class = np.argmax(x.data.cpu().numpy())
    print(target_class)
    # Target for backprop
    one_hot_output = torch.cuda.FloatTensor(1, x.size()[-1]).zero_()
    one_hot_output[0][target_class] = 1
    # Zero grads
    model.module.zero_grad()
    model.module.classifier.zero_grad()
    # Backward pass with specified target
    x.backward(gradient=one_hot_output, retain_graph=True)
    a.remove()
    b.remove()
    c.remove()
    d.remove()
    # Get hooked gradients
    guided_gradients1 = gradients1.data.cpu().numpy()[-1]
    # Get convolution outputs
    target1 = conv_output1[0]
    # Get weights from gradients
    weights1 = np.mean(guided_gradients1, axis=(1, 2))  # Take averages for each gradient
    # Create empty numpy array for cam
    cam1 = np.ones(target1.shape[1:], dtype=np.float32)
    # Multiply each weight with its conv output and then, sum
    for i, w in enumerate(weights1):
        cam1 += w * target1[i, :, :]
    cam1 = np.maximum(cam1, 0)
    cam1 = (cam1 - np.min(cam1)) / (np.max(cam1) - np.min(cam1))  # Normalize between 0-1
    cam1 = np.uint8(cam1 * 255)  # Scale between 0-255 to visualize
    cam1 = np.uint8(Image.fromarray(cam1).resize((x1.shape[2],
                   x1.shape[3]), Image.ANTIALIAS))/255
    cam1s.append(cam1)

    # Get hooked gradients
    guided_gradients2 = gradients2.data.cpu().numpy()[-1]
    # Get convolution outputs
    target2 = conv_output2[0]
    # Get weights from gradients
    weights2 = np.mean(guided_gradients2, axis=(1, 2))  # Take averages for each gradient
    # Create empty numpy array for cam
    cam = np.ones(target2.shape[1:], dtype=np.float32)
    # Multiply each weight with its conv output and then, sum
    for i, w in enumerate(weights2):
        cam += w * target2[i, :, :]
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
    cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
    cam = np.uint8(Image.fromarray(cam).resize((x2.shape[2],
                   x2.shape[3]), Image.ANTIALIAS))/255
    cam2s.append(cam)
#     print("processed tile ", i)


import matplotlib.cm as mpl_color_map
import copy

fig1, ax1 = plt.subplots(1,6, sharey=True, figsize=(30,5))
fig2, ax2 = plt.subplots(1,6, sharey=True, figsize=(30,5))

cams = [cam1s, cam2s]
inputs = [x1s, x2s]
axes = [ax1, ax2]
for k in range(2):
    for i in range(3):
        # Get colormap
        color_map = mpl_color_map.get_cmap('hsv')
        no_trans_heatmap = color_map(cams[k][i])
        # Change alpha channel in colormap to make sure original image is displayed
        heatmap = copy.copy(no_trans_heatmap)
        heatmap[:, :, 3] = 0.5
        heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
        no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))
        img = np.moveaxis(inputs[k][i][0], 0, 2)
        img = Image.fromarray((img*255).astype(np.uint8))
        # Apply heatmap on iamge
        heatmap_on_image = Image.new("RGBA", img.size)
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, img.convert('RGBA'))
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
        axes[k][i*2].imshow(img)
        axes[k][i*2+1].imshow(heatmap_on_image)
fig1.tight_layout()
fig1.savefig(path + "cam_map1.png")
fig2.tight_layout()
fig2.savefig(path + "cam_map2.png")

# ## 2. Plot performance

def roc_plot_compare(mean_tprs, auc_values, modelnames, colors, path, filename ):
    # plt.rcParams['font.sans-serif']=['Arial']
    plt.rcParams['axes.unicode_minus']=False 
#     plt.grid(linestyle = "--")
    ax = plt.gca()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    fig = plt.gcf()
    fig.set_size_inches(5.5, 4.5)
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(len(mean_tprs)):
        plt.plot(mean_fpr, mean_tprs[i], color=colors[i],
                     label=r'%s - ROC (AUC=%0.3f$\pm$%0.3f)' % (modelnames[i], np.mean(auc_values[i]), np.std(auc_values[i])),
                     lw=2, alpha=1)
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',label='Chance')

    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])

    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.legend(loc="lower right",fontsize=11)


    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(path + filename + '.svg',format='svg')

    plt.show()


#subnet test
a1 = np.load(project_path + 'results/subnet/roc_6class_test_mean_tpr_subnet.npy')
b1 = np.load(project_path + 'results/subnet/roc_6class_test_auc_values_subnet.npy')

#xception no-transfer test
a2 = np.load(project_path + 'results/xception/roc_6class_notrans_mean_tpr.npy')
b2 = np.load(project_path + 'results/xception/roc_6class_notrans_auc_values.npy')

#xception transfer test
a3 = np.load(project_path + 'results/xception/roc_6class_trans_mean_tpr.npy')
b3 = np.load(project_path + 'results/xception/roc_6class_trans_auc_values.npy')

#resnet50 test
a4 = np.load(project_path + 'results/resnet50/roc_6class_test_mean_tpr_resnet50.npy')
b4 = np.load(project_path + 'results/resnet50/roc_6class_test_auc_values_resnet50.npy')

#bcnn test
a5 = np.load(project_path + 'results/bcnn/roc_6class_test_mean_tpr.npy')
b5 = np.load(project_path + 'results/bcnn/roc_6class_test_auc_values.npy')


roc_plot_compare([a1,a2,a3,a4,a5], [b1,b2,b3,b4,b5], ['Subnet', 'Xception','Xception (TL)','Resnet50','BCNN'],  ['aqua','darkorange','blue','yellow', 'green'], path, 'auc_compared_conference')


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
    names.append(resolution_name)
    
#the last one is image patches used in the paper
tprs.append(np.load(path + 'roc_6class_test_mean_tpr.npy'))
aucs.append(np.load(path + 'roc_6class_test_auc_values.npy'))
names.append('256x256/512x512')    
roc_plot_compare(tprs, aucs, names,  ['darkorange','yellow','aqua', 'blue', 'red', 'green'],  path, 'auc_compared_resolution_conference')


classesnames = ['Necrosis Palisading', 'Microvascular Proliferation', 'Blood Vessel', 'Necrosis Geographic', 'Brain Tissue', 'Tumor Background']
def roc_plot(n_classes_, y_tests_,y_prediction_proba_, path, filename ):
    # plt.rcParams['font.sans-serif']=['Arial']
    plt.rcParams['axes.unicode_minus']=False 
#     plt.grid(linestyle = "--")
    ax = plt.gca()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    fig = plt.gcf()
    fig.set_size_inches(5.5, 4.5)

    tprs_all = []
    aucs_all = []
    mean_fpr = np.linspace(0, 1, 100)

    auc_values = []
    colors = ['red', 'blue', 'cyan','yellow','lightgrey','green','green']
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
                     label=r'%d. %s - ROC (AUC=%0.3f)' % (j+1,classesnames[j], mean_auc), linestyle='--',
                     lw=1., alpha=1)
            auc_values.append(mean_auc)
            
        else:
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, color=colors[j],
                     label=r' %d. %s - Mean ROC (AUC=%0.3f$\pm$%0.3f)' % (j+1, classesnames[j], mean_auc, std_auc),linestyle='--',
                     lw=1., alpha=1)
            auc_values.append(mean_auc)

    mean_tpr = np.mean(tprs_all, axis=0)
    mean_tpr[0] = .0
    mean_tpr[-1] = 1.0
    
    if n_classes_ > 1:
        plt.plot(mean_fpr, mean_tpr, color=colors[-1],
                 label=r'Mean ROC (AUC=%0.3f$\pm$%0.3f)' % (np.mean(auc_values),  np.std(auc_values)),
                 lw=2, alpha=1)
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',label='Chance', alpha=1)



    std_tpr = np.std(tprs_all, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.1,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([0., 1.0])
    plt.ylim([0., 1.0])

    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.legend(loc="lower right",fontsize=10)


    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(path + filename + '.svg',format='svg')

    plt.show()
    return mean_tpr, auc_values


# the test roc/auc
yts=[]
yts.append(y_test)
pbs=[]
pbs.append(probas_)
mean_tpr, auc_values = roc_plot(6,yts,pbs, path, 'roc_6class_test_conference')

