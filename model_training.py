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

#feature importances
RFC = RandomForestClassifier(criterion='entropy',max_depth=30,max_features='sqrt',n_estimators=400).fit(data_X,data_y)
import matplotlib.pyplot as plt
importances = RFC.feature_importances_
feat_names = data_X.columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature importances by RandomForest")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()

#adding top20
scoring = ['f1']
f1_dict = {}
for i in range(1,40):
    use_features = feat_names[indices][:i]
    RFC = RandomForestClassifier(criterion='entropy',max_depth=30,max_features='sqrt',n_estimators=400)
    scores  = cross_validate(RFC,data_X[use_features],data_y,cv = 5,scoring = scoring)
    f1_dict[i] = scores["test_f1"].mean()
import matplotlib.pyplot as plt
plt.plot(f1_dict.keys(),f1_dict.values())
plt.show()

#Adaboost
DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto",max_depth = None)
param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [10, 50, 100, 500],
              'learning_rate' : [0.0001, 0.001, 0.01, 0.1, 1.0]
             }
CV_abc = GridSearchCV(estimator=ABC, param_grid=param_grid, cv= 10,n_jobs=-1,scoring = 'f1')
CV_abc.fit(data_X,data_y)
CV_abc.best_params_

#SVM
from sklearn.svm import SVC

SVM = SVC()
tuned_parameters = [{'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100]}]
CV_svm = GridSearchCV(estimator=SVM, param_grid=tuned_parameters, cv= 5,n_jobs=-1,scoring = 'f1')
CV_svm.fit(data_train_top20,data_y_top20)
CV_svm.best_params_

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
k_range = list(range(1,100,5))
param_grid = dict(n_neighbors=k_range)
CV_knn = GridSearchCV(knn, param_grid, cv=5, scoring='f1')
CV_knn.fit(data_train_top20,data_y_top20)
CV_knn.best_params_

#LightGBM
import lightgbm as lgb

lg = lgb.LGBMClassifier(random_state=11,silent=False)
param_dist = {"max_depth": [10,20,30,40,50],
              "learning_rate" : [0.001,0.01,0.05,0.1,0.5],
              "num_leaves": [50,100,200,300,350,500],
              "n_estimators": [100,200,300,400,500,600,700]
             }

grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="f1", verbose=1)
grid_search.fit(temp_train_X[use_features],temp_train_y)
print(grid_search.best_params_,grid_search.best_score_)

#catboost
import catboost as cb

params = {#'depth': [2,4, 7, 10,15,20],
          'learning_rate' : [0.15],
         'l2_leaf_reg': [4],
         'iterations': [750,800,850,900,950,1000]}
cb = cb.CatBoostClassifier()
cb_model = GridSearchCV(cb, params, scoring="f1", cv = 3)
cb_model.fit(data_X,data_y)
print(cb_model.best_params_,cb_model.best_score_)

#xgboost
import xgboost as xgb
# A parameter grid for XGBoost
xgb_model = xgb.XGBClassifier()

params = {
        'min_child_weight': [1,  10],
        'gamma': [0.5, 1, 2, 5],
        'colsample_bytree': [0.6, 1.0],
        'max_depth': [3, 5,None]
        }

CV_xgb = GridSearchCV(xgb_model, params, n_jobs=5, 
                   cv=5, 
                   scoring='f1')
CV_xgb.fit(data_X[use_features],data_y)
CV_xgb.best_params_

#Decisiontree
parameters={'min_samples_split' : range(10,500,50),'max_depth': range(1,20,4)}
DT=DecisionTreeClassifier()
CV_dt=GridSearchCV(DT,parameters,cv=5,scoring='f1')
CV_dt.fit(data_train_X_one[use_features_onehot],data_train_y_one)
CV_dt.best_params_

#NN
from sklearn.neural_network import MLPClassifier
NN = MLPClassifier()
parameters = {'solver': ['lbfgs'], 'max_iter': [100,500,1000,1500,1800,2000,2200,2500]}
CV_nn = GridSearchCV(NN, parameters, cv = 5,n_jobs=-1,scoring='f1')
CV_nn.fit(data_train_X_one[use_features_onehot],data_train_y_one)
CV_nn.best_params_

#Majority vote
#top20 features use ordinal encode
DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto",max_depth = None,criterion = 'entropy',splitter= 'best')
ABC = AdaBoostClassifier(base_estimator = DTC, learning_rate = 0.1, n_estimators = 90)

RFC = RandomForestClassifier(criterion='entropy',max_depth=30,max_features='sqrt',n_estimators=300)

XGB = xgb.XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

import statistics
final_dict = {}
for i in (range(5)):
    X_train,X_test,y_train,y_test = train_test_split(data_X[use_features],data_y,train_size=0.8)

    ABC.fit(X_train,y_train)
    XGB.fit(X_train,y_train)
    RFC.fit(X_train,y_train)

    pred1=ABC.predict(X_test)
    pred2=XGB.predict(X_test)
    pred3=RFC.predict(X_test)

    final_pred = np.array([])
    for j in range(0,len(X_test)):
        final_pred = np.append(final_pred, statistics.mode([pred1[j], pred2[j], pred3[j]]))
    final_dict[i] = f1_score(y_test,final_pred)

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