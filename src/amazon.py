from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV
import time
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB

enc = OneHotEncoder()
model = LogisticRegression(C = 1, penalty = 'l2', intercept_scaling = 0.01, solver = 'liblinear', class_weight = 'balanced', n_jobs = -1)
gnb = GaussianNB()

train_df = pd.read_csv('/Users/admin/Dropbox/EE379K/project/ee379k_project/dataset/train.csv')
test_df = pd.read_csv('/Users/admin/Dropbox/EE379K/project/ee379k_project/dataset/test.csv')
trainSet = train_df.as_matrix()
testSet = test_df.as_matrix()
trainX = trainSet[:, 1:]
trainY = trainSet[:, 0]
testX = testSet[:, 1:]
#testY = testSet[:, 0]

##Initial Testing with SGDC##
#X_t, X_val, y_t, y_val = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
#linear_clf = SGDClassifier()
#linear_clf.fit(X_t,y_t)
#linear_predictions = linear_clf.predict(X_val)
#print "Linear Predictions"
#print linear_predictions
#print "Y_val"
#print y_val
#print "RocAucScore for Linear"
#print metrics.roc_auc_score(linear_predictions,y_val)
##Initial Testing with SGDC##

##GNB - model not worth it###
##y_pred= gnb.fit(trainX,trainY)
##print "Number of mislabeled points"
##print trainY != y_pred
##print (trainY != y_pred).sum()
##GNB- nodel not worth it##

##ONE HOT ENCODING
enc.fit(np.concatenate((trainX, testX)))
trainX = enc.transform(trainX)
testX = enc.transform(testX)
###ONE HOT ENCODING

X_t, X_val, y_t, y_val = train_test_split(trainX,trainY,test_size=0.33,random_state=42)

###XGB###
print "Prediction XGB"
##Initial: gbm = xgb.XGBClassifier(max_depth=5, n_estimators=1000, learning_rate=0.05)
gbm = xgb.XGBClassifier(max_depth=100, n_estimators=200, colsample_bytree=0.3)
gbm.fit(trainX, trainY)
##predictionsXGB = gbm.predict(testX)
##print predictionsXGB

##trainEstXGB= gbm.predict_proba(trainX)
##trainEstXGB= trainEstXGB[:,1]
testEstXGB = gbm.predict_proba(testX)
testEstXGB=testEstXGB[:,1]
subm = pd.DataFrame({'id':test_df['id'], 'ACTION':testEstXGB})
subm.to_csv('/Users/admin/Dropbox/EE379K/project/ee379k_project/submissions/submissionXGB_optimized.csv', index = False)
print "Saved results at submissionXGB_optimized.csv"
###XGB###

##SGDC CLASSIFIER TESTING##
#linear_clf = SGDClassifier()
#linear_clf.fit(trainX,trainY)
#linear_predictions = linear_clf.predict(testX)
#print linear_predictions
#print roc_auc_score(linear_predictions,testX)
##SGDC CLASSIFIER TESTING##


#Logistic Regression#
print "predicting Logistic Regression"
model.fit(trainX,trainY)
trainEst = model.predict_proba(trainX)
trainEst = trainEst[:,1]
testEst = model.predict_proba(testX)
testEst = testEst[:, 1]
timestamp= str(int(time.time()))
submission = pd.DataFrame({'id':test_df['id'], 'ACTION':testEst})
submission.to_csv('/Users/admin/Dropbox/EE379K/project/ee379k_project/submissions/submission_' + timestamp + '.csv', index = False)
print "Saved results at submission_*newtimestamp*.csv"

##fpr, tpr, thresholds = metrics.roc_curve(trainY, trainEst, pos_label = 1)
##auc = metrics.auc(fpr, tpr)
##print "AUC SCORE %f" %(auc)

