import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from fastFM import als,sgd,mcmc
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics

#num iterations 50
#num factors 25
#fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
#fm = sgd.FMClassification(n_iter=1000, init_stdev=0.1, l2_reg_w=0,l2_reg_V=0, rank=2, step_size=0.1)


train_df = pd.read_csv('/Users/admin/Dropbox/EE379K/project/ee379k_project/dataset/train.csv')
test_df = pd.read_csv('/Users/admin/Dropbox/EE379K/project/ee379k_project/dataset/test.csv')
trainSet = train_df.as_matrix()
testSet = test_df.as_matrix()
trainX = trainSet[:, 1:]
trainY = trainSet[:, 0]
testX = testSet[:, 1:]

##ONE HOT ENCODER##
enc = OneHotEncoder()
enc.fit(np.concatenate((trainX, testX)))
trainX = enc.transform(trainX)
testX = enc.transform(testX)
X_t, X_val, y_t, y_val = train_test_split(trainX,trainY,test_size=0.33,random_state=42)
##ONE HOT ENCODER##

##INITIAL TEST OF MCMC WITHOUT OPTIMIZATION##
#fm.fit(trainX, trainY)
#testY = fm.predict(testX)
#print(testY)
###y_pred = fm.fit_predict(trainX, trainY, testX)
###y_pred_proba = fm.fit_predict_proba(trainX, trainY, testX)
##INITIAL TEST OF MCMC WITHOUT OPTIMIZATION##

fm = None
#TESTS
#n_iter = [10,20,30,40,50,60,70,80,90, 100,500,1000,2000,3000]
#rank_iter = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,50]
#stdev_iter= [0,0.05,0.1,0.5,1.0]
#for i in stdev_iter:
##TESTS
fm = mcmc.FMClassification(n_iter=80, rank=25, init_stdev=0.1)
y_pred = fm.fit_predict(trainX, trainY, testX)
y_pred_proba = fm.fit_predict_proba(trainX, trainY, testX)
y_pred_proba_auc = fm.fit_predict_proba(X_t, y_t, X_val)

fpr,tpr, thresholds = metrics.roc_curve(y_val, y_pred_proba_auc, pos_label = 1)
auc = metrics.auc(fpr, tpr)
#print auc
print "AUC: " %(auc)


#submission = pd.DataFrame({'id':test_df['id'], 'ACTION':y_pred_proba_auc})
#submission.to_csv('/Users/admin/Dropbox/EE379K/project/ee379k_project/submissions/submission_FMoptimized.csv', index = False)
print "Saving results at submission_FMoptimized.csv"


