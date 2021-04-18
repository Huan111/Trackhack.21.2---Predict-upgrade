import numpy as np
import pandas as pd

teamname = 'emotional-support-vector-machine-unsw'
root_folder='s3://tf-trachack-notebooks/'+teamname+'/jupyter/jovyan/'

data_train = pd.read_csv(root_folder+"guohuan-li/new_data/dev_final_merge.csv")
data_val = pd.read_csv(root_folder+"guohuan-li/new_data/eval_final_merge.csv")

#drop some features not in both datasets
train_lst = list(data_train.columns[3:])
val_lst = list(data_val.columns[1:])
drop_lst = np.setdiff1d(train_lst,val_lst)
data_train.drop(drop_lst, axis=1,inplace=True)
train_lst = list(data_train.columns[3:])
val_lst = list(data_val.columns[1:])
drop_lst = np.setdiff1d(val_lst,train_lst)
data_val.drop(drop_lst, axis=1,inplace=True)

#extract the training data
data_y = data_train['upgrade'].replace({'no':0,'yes':1})
data_X = data_train.drop(['line_id','upgrade_date','upgrade'],axis = 1)
data_val_X = data_val.drop(['line_id'],axis = 1)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

DTC = DecisionTreeClassifier()
RFC = RandomForestClassifier()
ABC = AdaBoostClassifier()
LR = LogisticRegression(max_iter=500)
MLP = MLPClassifier(max_iter = 500)
SVM = SVC()

clfs = [DTC,RFC,ABC,LR,MLP,SVM]
names = ['DTC','RFC','ABC','LR','MLP','SVM']

from sklearn.model_selection import cross_validate
scoring = ['f1','precision','recall','accuracy']

#RFC
from sklearn.model_selection import GridSearchCV
param_grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
 'criterion' :['gini', 'entropy']}
CV_rfc = GridSearchCV(estimator=RFC, param_grid=param_grid, cv= 10,n_jobs=-1,scoring = 'f1')
CV_rfc.fit(data_X,data_y)
CV_rfc.best_params_

#Adaboost
param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [10, 50, 100, 500],
              'learning_rate' : [0.0001, 0.001, 0.01, 0.1, 1.0]
             }
CV_abc = GridSearchCV(estimator=ABC, param_grid=param_grid, cv= 10,n_jobs=-1,scoring = 'f1')
CV_abc.fit(data_X,data_y)
CV_abc.best_params_


################################top20 features in Random Forests using LR$########################
from category_encoders import BinaryEncoder
import pandas as pd
teamname = 'emotional-support-vector-machine-unsw'
root_folder='s3://tf-trachack-notebooks/'+teamname+'/jupyter/jovyan/'

data_train_one_top20 = pd.read_csv(root_folder+"guohuan-li/new_data_1/train_top20.csv")
data_eval_one_top20 = pd.read_csv(root_folder+"guohuan-li/new_data_1/eval_top20.csv")

use_features = ['net_work_count','red_count',
 'net_sms_mean_sum',
 'net_voice_min_mean_sum',
 'sus_count',
 'de_re_counts',
 'red_mean_rev',
 'net_voice_count_mean_sum',
 'net_mms_mean_sum',
 'net_work_mean_kb',
 'net_sms_ratio',
 'net_voice_count_ratio',
 'net_voice_min_ratio',
 'net_mms_ratio',
 'gsma_model_name',
 'internal_storage_capacity',
 'channel_unique',
 'channel_most_fre',
 'total_ram',
 'year_released']

#
data_train = data_train_one_top20[use_features]
data_val = data_eval_one_top20[use_features]
data_y = data_train_one_top20['upgrade']

cat_features = ['gsma_model_name','internal_storage_capacity','channel_most_fre','total_ram','year_released']

data_train = BinaryEncoder(cols=cat_features).fit_transform(data_train)
data_val = BinaryEncoder(cols=cat_features).fit_transform(data_val)

#Logistic regression
param_grid = {
    'penalty' : ['l1', 'l2'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['liblinear','lbfgs']}
LR = LogisticRegression()
CV_LR = GridSearchCV(estimator=LR, param_grid=param_grid, cv= 5,verbose=True,n_jobs=-1,scoring = 'f1')
CV_LR.fit(data_train,data_y)
print(CV_LR.best_params_)
print(CV_LR.best_score_)