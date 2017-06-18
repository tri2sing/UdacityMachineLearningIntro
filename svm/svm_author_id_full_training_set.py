#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
#########################################################
clf = SVC(kernel='rbf', C=10000) #The value for C is from experiments
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s",
t1 = time()
labels_pred = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s",
accuracy = accuracy_score(labels_pred, labels_test)
print "accuracy = ", accuracy
for i in [10, 26, 50]:
    print "label ", i, "= ", labels_pred[i]
print "Number of Chris emails = ", sum(x for x in labels_pred if x == 1)
