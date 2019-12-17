# Author: Guoqing Bao
# School of Computer Science, The University of Sydney
# Date: 2019-12-12
# GitHub Project Link: https://github.com/guoqingbao/Pathofusion
# Please cite our work if you found it is useful for your research or clinical practice

#%%
# from IPython import get_ipython

#%%
from PIL import Image
# make sure we can process large image
Image.MAX_IMAGE_PIXELS = 2000000000 
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
import warnings;
warnings.filterwarnings('ignore');
import os
import cv2
# get_ipython().run_line_magic('matplotlib', 'inline') 
import matplotlib.pyplot as plt
import pandas as pd
import openslide
import sqlite3
import MySQLdb
import io
from urllib.request import urlopen

# This module is used to extract image patches from H&E
# Make sure your labelling website is settled, which include labelling coordinates for your H&E or IHC images.
#%%

# extract image patches from IHC images is pretty similar like this (H&E), except the saving path, types,reading different tables from database

mydb = MySQLdb.connect(
  host="xx.xx.xx.xx", #  the labelling website IP you hosted, replace with your IP
  user="root",
  passwd="xxx", # database password corresponding to the labelling website, replace with your password
  db="WebLabelling"
)
# mydb.close()

project_path = "./"

#%%
def sqlite_create_or_open_db(db_file):
    db_is_new = not os.path.exists(db_file)
    conn = sqlite3.connect(db_file)
    if db_is_new:
        print('Creating schema')
        sql = '''create table if not exists PICTURES(
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        PATIENT_ID INTEGER,
        TYPE INTEGER,
        GRADE INTEGER,
        ORIGIN_SIZE INTEGER,
        SIZE INTEGER,
        X INTEGER,
        Y INTEGER,
        PICTURE BLOB     
        );'''
        conn.execute(sql) # shortcut for conn.cursor().execute(sql)
    else:
        print('Schema exists\n')
    return conn

def sqlite_insert_picture(conn, patient_id, tp, grade, origin_sz, sz,x, y, image):
    sql = "INSERT INTO PICTURES (PATIENT_ID, TYPE, GRADE, ORIGIN_SIZE, SIZE,X, Y, PICTURE) VALUES(?, ?, ?, ?, ?, ?, ?, ?)"
#     print(sql)
    conn.execute(sql,[patient_id, tp, grade, origin_sz, sz, x, y, sqlite3.Binary(image.tobytes())]) 
    conn.commit()   
    
def get_pictures(conn, patient_id):
    if patient_id == -1:
        sql = "SELECT ID, PATIENT_ID, TYPE  FROM PICTURES"
    else:
        sql = "SELECT ID, PATIENT_ID, TYPE FROM PICTURES WHERE PATIENT_ID = "+ str(patient_id) 
    return pd.read_sql_query(sql, conn) 

def del_picture(conn, patient_id, x, y):
    conn.execute("delete from PICTURES where PATIENT_ID=" + str(patient_id) + " and X="+str(x) +" and Y="+str(y)) 
    conn.commit() 
    
def get_labeling_data(conn, patient_id):
    if patient_id == -1:
        sql = "SELECT *  FROM imagelist_imagelist WHERE labels != 'null' and labels != '[]'"
    else:
        sql = "SELECT * FROM imagelist_imagelist WHERE labels != 'null' and labels != '[]' and pid = "+ str(patient_id) 
    return pd.read_sql_query(sql, conn) 

def get_labeling_data_before(conn, patient_id):
    if patient_id == -1:
        sql = "SELECT *  FROM imagelist_imagelist WHERE labels != 'null' and labels != '[]'"
    else:
        sql = "SELECT * FROM imagelist_imagelist WHERE labels != 'null' and labels != '[]' and pid < "+ str(patient_id) 
    return pd.read_sql_query(sql, conn) 

def get_labeling_data_after(conn, patient_id):
    if patient_id == -1:
        sql = "SELECT *  FROM imagelist_imagelist WHERE labels != 'null' and labels != '[]'"
    else:
        sql = "SELECT * FROM imagelist_imagelist WHERE labels != 'null' and labels != '[]' and pid >= "+ str(patient_id) 
    return pd.read_sql_query(sql, conn) 

def sqlite_create_labels_table(conn):
    sql = '''create table if not exists LABELS(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    PATIENT_ID INTEGER,
    THUMB_IMAGE TEXT,
    IMAGE TEXT,
    LABELS TEXT    
    );'''
    conn.execute(sql) # shortcut for conn.cursor().execute(sql)
    return conn
def update_label_in_mysqldb(conn, patient_id, labels):
    cur = conn.cursor()
    cur.execute("update imagelist_imagelist set imagelist_imagelist.labels = %s where imagelist_imagelist.pid = %s", (labels,patient_id))
    conn.commit() 
    results =cur.fetchall()
    print(results)
    
def get_images(conn, imageids):

    ids = ""
    for i in imageids:
        ids = ids + str(i) +","
    ids = ids[:-1]
    
    sql = "SELECT ID, TYPE, PICTURE, PATIENT_ID FROM PICTURES WHERE TYPE != 7 and ID in (" + ids + ") ORDER BY " 
    
    sql = sql + "CASE ID "
    i = 1
    for id in imageids:
        sql = sql + " WHEN " + str(id) + " THEN " + str(i)
        i = i + 1

    sql += " END"

    d = pd.read_sql_query(sql, conn)
    return d 

def get_images_by_position(conn, x1, x2, y1, y2):
    
    sql = "SELECT ID, TYPE, PICTURE, PATIENT_ID FROM PICTURES WHERE (X BETWEEN "+ str(x1) + " AND "+str(x2)+") AND (Y BETWEEN "+ str(y1) + " AND "+str(y2)+")"   

    d = pd.read_sql_query(sql, conn)
    return d 


#%%
# we read all labelling coordinates from the database (website backend)
frame = get_labeling_data_after(mydb, 0)


#%%
# we calculate how many labelling coordinates in each pathology image.
import json
dots = []
for index, row in frame.iterrows():
    obj = json.loads(row.labels.replace("'","\"")) 
    dots.append(len(obj))


#%%
frame.ix[:,"dots"] = dots


#%%
# we create the databases that to save image patches and corresponding labelling type
conn_64 = sqlite_create_or_open_db(project_path + "data/brain_labeling_64.db")
conn_64 = sqlite_create_labels_table(conn_64)
conn_128= sqlite_create_or_open_db(project_path + "data/brain_labeling_128.db")
conn_128 = sqlite_create_labels_table(conn_128)


#%%
# you may choose 256x256 and 512x512 only, which is the most effective patch pair in our study for GBM pathology analysis
conn_256 = sqlite_create_or_open_db(project_path + "data/brain_labeling_256.db")
conn_256 = sqlite_create_labels_table(conn_256)
conn_512= sqlite_create_or_open_db(project_path + "data/brain_labeling_512.db")
conn_512 = sqlite_create_labels_table(conn_512)


#%%
#exclude some pathology images you don't want to remain
excluded = []
newFrame = frame.loc[~frame.pid.isin(excluded)]


#%%
newFrame


#%%
import json
dots = []
for index, row in newFrame.iterrows():
    obj = json.loads(row.labels.replace("'","\"")) 
    dots.append(len(obj))


#%%
newFrame.ix[:,'dots'] = dots


#%%
import json

# make sure you uploaded your pathology images into below path, replace xx.xx.xx.xx with your website IP
image_path = "http://xx.xx.xx.xx/static/labelling/"

# the pathology types we used for GBM, you can define your own types for other pathology images.
#1 Necrosis-palisading, 2 MicVas-Proliferation, 3 Blood-Vessel, 4 Necrosis-Geo, 5 Brain-Tissue, 6 Tumor, 7 Satellitosis

# we used 6 types of GBM features
color_types = {1:'black', 2:'yellow', 3:'blue', 4:'cyan',5:'grey', 6:'green'}
GRADE = 4

# we iterate the labelling datasets and crop the whole-slide images based on expert markings
fd=None
for index, row in newFrame.iterrows():
    obj = json.loads(row["labels"].replace("'","\""))
    old_row = None
    obj_old = None
    
    if fd!=None:
        old_row = fd[fd["PATIENT_ID"]==row["pid"]]
    
    if old_row!=None and old_row.count()["PATIENT_ID"] > 0:
        obj_old = json.loads(old_row.iloc[0,3].replace("'","\""))
    else:
        # you may choose to use 128x128 and 256x256 only, which is proved in our paper as the most effective resolutions for GBM pathology analysis
        sql = "INSERT INTO LABELS (PATIENT_ID, THUMB_IMAGE, IMAGE, LABELS) VALUES(?, ?, ?, ?)"
        
        conn_64.execute(sql,[row["pid"], row["thumb_image"], row["image"], row["labels"]]) 
        conn_64.commit()
        conn_128.execute(sql,[row["pid"], row["thumb_image"], row["image"], row["labels"]]) 
        conn_128.commit()
        conn_256.execute(sql,[row["pid"], row["thumb_image"], row["image"], row["labels"]]) 
        conn_256.commit()
        conn_512.execute(sql,[row["pid"], row["thumb_image"], row["image"], row["labels"]]) 
        conn_512.commit()
        
        print("new records pid=",row["pid"])
    # open the whole-slide image
    urlsession = urlopen(image_path + row["image"])
    image_file = io.BytesIO(urlsession.read())    
    img_origin = Image.open(image_file)
    
    for key, value in color_types.items():
        label = [[item['x'], item['y']] for item in obj if item['style']==value] 
        label_del = []
        if obj_old != None:
            label_old = [[item['x'], item['y']] for item in obj_old if item['style']==value]
            k = np.array(label)
            k2 = np.array(label_old)

            c = k[np.in1d(k.view(dtype='f8,f8').reshape(k.shape[0]),k2.view(dtype='f8,f8').reshape(k2.shape[0]))]
#             c = k[np.in1d(list(map(np.ndarray.dumps, k)), list(map(np.ndarray.dumps, k2)))]
#             print(c)
            label = [lb for lb in k if lb not in c]
            label_del = [lb for lb in k2 if lb not in c]
#         print(label)

        if len(label_del)>0:
            print("del images for case ", row["pid"], " :", label_del)
        for i in range(len(label_del)):
            p = label[i]
            del_pictures(row["pid"], p[0], p[1])

        # crop the whole-slide image as image patches    
        if len(label) >0:
            print("crop image " + row["image"] + ", color " + value + ", len " + str(len(label)))            
        for i in range(len(label)):
            p = label[i]
            extend = 512/2
            img = img_origin.crop((p[0] - extend,p[1] - extend,p[0]-extend+512,p[1]-extend+512)) # (left, upper, right, lower)-tuple.
            sqlite_insert_picture(conn_512,row['pid'],key, GRADE, 512, 512, p[0], p[1], img)
            extend = 256/2
            img = img_origin.crop((p[0] - extend,p[1] - extend,p[0]-extend+256,p[1]-extend+256)) # (left, upper, right, lower)-tuple.
            sqlite_insert_picture(conn_256, row['pid'], key, GRADE, 256, 256, p[0], p[1], img)
            
            extend = 128/2
            img = img_origin.crop((p[0] - extend,p[1] - extend,p[0]-extend+128,p[1]-extend+128)) # (left, upper, right, lower)-tuple.
            sqlite_insert_picture(conn_128,row['pid'],key, GRADE, 128, 128, p[0], p[1], img)
            extend = 64/2
            img = img_origin.crop((p[0] - extend,p[1] - extend,p[0]-extend+64,p[1]-extend+64)) # (left, upper, right, lower)-tuple.
            sqlite_insert_picture(conn_64, row['pid'], key, GRADE, 64, 64, p[0], p[1], img)
            
    img_origin.close()


#%%
# let's see the statistics of the image patche dataset
frame = get_pictures(conn_512, -1)


#%%
frame.describe()


#%%
counts = frame.TYPE.value_counts().sort_index().tolist()
counts


#%%
# you can also plot the statistics
colors = []

colors.append([1,0,0]) #red necrosis palisading
colors.append([0,0,1]) #blue MicVas-Proliferation
colors.append([0,1,1]) #cyan blood-vessel

colors.append([1,1,0]) #yellow Necrosis-Geo
colors.append([1,1,1])# white Magenta  5 Brain-Tissue

colors.append([0,0.7,0.7]) #green tumor background

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(aspect="equal"))

pie_chart_exploded = (0.06, 0.06, 0.06, 0.06, 0.06, 0.06)
wedges, texts, autotexts = ax.pie(counts[:6]  , autopct='%1.1f%%', explode=pie_chart_exploded,  shadow=True, startangle=90, 
                                  colors=colors,
                                  textprops=dict(color="black"))
labels = ["Necrosis Palisading","MicVas-Proliferation","Blood Vessel","Necrosis-Geo","Brain Tissue","Tumor Background"]
plt.legend(wedges, labels, bbox_to_anchor=(1.,0.5), loc="center right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)
plt.setp(autotexts, size=12, weight="bold")
    
ax.set_title("Data Distribution",size=12, weight="bold")
plt.savefig(project_path + 'results/data_distribution.svg',format='svg')
plt.show()


#%%



