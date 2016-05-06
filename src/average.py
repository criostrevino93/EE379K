from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_iris, load_digits, load_boston
import time
import xgboost as xgb
from sklearn.metrics import roc_auc_score


linearReg = pd.read_csv('/Users/admin/Dropbox/EE379K/project/ee379k_project/submissions/submission_1462240590.csv', header=0, names=['ACTION', 'ID'])
fm=  pd.read_csv('/Users/admin/Dropbox/EE379K/project/ee379k_project/submissions/submissionXGB_optimized.csv',header=0, names=['ACTION', 'ID'])
xgbDF = pd.read_csv('/Users/admin/Dropbox/EE379K/project/ee379k_project/submissions/submissionXGB_optimized.csv',header=0, names=['ACTION', 'ID'])

rf= pd.read_csv('/Users/admin/Downloads/RFNI250.csv',header=0, names=['ACTION', 'ID'])
#linearSet= linearReg.as_matrix()
#fmSet=fm.as_matrix()

#linearAction=linearSet[0]
#fmAction=fmSet[:,0:]
#print "Linear"
#print linearSet

linearRegAction= linearReg.drop(['ID'], axis=1)
fmAction = fm.drop(['ID'], axis=1)
xgbAction = xgbDF.drop(['ID'], axis=1)
rfAction = rf.drop(['ID'], axis=1)

#print linearRegAction
#print linearReg

#or i in range(0,58919):
#df_concat = pd.concat((linearRegAction,fmAction))
#print df_concat
#by_row_index = df_concat.groupby(df_concat.index)
#df_means = by_row_index.mean()

#print df_means

average = pd.DataFrame((xgbAction.values + rfAction.values)/2, fmAction.index, fmAction.columns)
#print average




#gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
#predictions = gbm.predict(test_X)



submission = pd.DataFrame({'id':linearReg['ID'], 'ACTION':average['ACTION']})
submission.to_csv('/Users/admin/Dropbox/EE379K/project/ee379k_project/submissions/submission_optimizedXGBrf.csv', index = False)