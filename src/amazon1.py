#!/usr/bin/env python 
import numpy as np
from sklearn import (metrics, cross_validation, linear_model, ensemble, preprocessing)
from sklearn.svm import SVC

SEED = 42  # always use a seed for randomized procedures

#-------Fit models 
#model = linear_model.LogisticRegression(C=3)  
#model = ensemble.RandomForestClassifier()
#model = ensemble.ExtraTreesClassifier()
#model = ensemble.RandomForestClassifier(n_estimators=100)
#model = ensemble.RandomForestClassifier(n_estimators=250)

#-------Support Vector Machines (SVMs)
#model = SVC(kernel='rbf', probability=True)
#model = SVC(kernel='linear', probability=True)

#-------load data; x&Y are numpy arrays
print "loading data"
X = np.loadtxt(open('train.csv'), delimiter=',', usecols=range(1,9), 
skiprows=1)
y = np.loadtxt(open('train.csv'), delimiter=',', usecols=[0], skiprows=1)	
X_test = np.loadtxt(open('test.csv'), delimiter=',',usecols=range(1,9), 
skiprows=1)
y_test = np.zeros(X_test.shape[0])


#-------one-hot encodinig on training and test set category IDs
encoder = preprocessing.OneHotEncoder()
encoder.fit(np.vstack((X, X_test)))
X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
X_test = encoder.transform(X_test)

#-------Training & Stats
mean_auc = 0.0
n = 10  # repeat the CV procedure 10 times to get more precise results
for i in range(n):
        # for each iteration, randomly hold out 20% of the data as CV set
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y,test_size=.20, random_state=i*SEED)

        # train model and make predictions
        model.fit(X_train, y_train) 
        preds = model.predict_proba(X_cv)[:, 1]

        # compute AUC metric for this CV fold
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
        roc_auc = metrics.auc(fpr, tpr)
        print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
        mean_auc += roc_auc

print "Mean AUC: %f" % (mean_auc/n)

#-------Predictions
model.fit(X,y)
predictions = model.predict_proba(X_test)[:,1]
#*******RENAME FILE ACCORDING TO WHAT YOU RUN
filename = "RFNI2.csv"
with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
                f.write("%d,%f\n" % (i + 1, pred))

