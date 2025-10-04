from sklearn import cross_validation
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
import sys
sys.path.insert(0, '..')  

NUM_FOLDS = 10

csv_path = '/data/raw_data_repository/pneumonia_detection/stage1_image_bbox_full.csv'
df_all = pd.read_csv(csv_path)
name_list = list(set(df_all['patientId'].values.tolist()))

name_list = np.array(name_list)

kf = KFold(len(name_list), shuffle=True, n_folds=NUM_FOLDS, random_state=4242)

for num_fold, (train_index, test_index) in enumerate(kf):
    os.makedirs('/data/VPS/VPS_04/pneumonia_detection/pneumonia_ensemble/data/fold_10/fold'+str(num_fold), exist_ok=True)
    c_train,c_val = name_list[train_index].tolist(), name_list[test_index].tolist()
    out_train = open('/data/VPS/VPS_04/pneumonia_detection/pneumonia_ensemble/data/fold_10/fold'+str(num_fold) +'/train.txt','w')
    out_val = open('/data/VPS/VPS_04/pneumonia_detection/pneumonia_ensemble/data/fold_10/fold'+str(num_fold) +'/val.txt','w')
    print(len(c_train), len(c_val))
    for i in c_train:
        out_train.write(''.join(i)+'\n')
    for i in c_val:
        out_val.write(''.join(i)+'\n')
#         df_all[df_all['patientId'].isin(c_val)].groupby('class').size().plot.bar()

