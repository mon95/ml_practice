"""
	Program to actually run the ML algorithms and evaluate the performance
"""

from sklearn.cross_validation import *
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.linear_model import *
import numpy as np
from time import time
import pickle
import pandas as pd
import warnings

# import all the relevant modules

warnings.filterwarnings("ignore")		#silence annoying warnings

pos_vecs = np.load('pos_vecs.npy')						#load positive vectors
neg_vecs = np.load('neg_vecs.npy')[:len(pos_vecs)]		#load negative vectors
														#we discard extra negative vectors to keep the dataset balanced


train_x = np.append(pos_vecs, neg_vecs, axis = 0)		#We combine pos_vecs and neg_vecs together
														

train_y = np.array([1] * len(pos_vecs) + [0] * len(neg_vecs))		#generate labels corresponding to the vecs in train_x 

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size = 0.5, random_state = 0)

#The data is split into two parts, one for model training and another for model validation.
#The split ratio is 50 : 50

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)	#scale the data be subtracting the mean and dividing by std_dev
test_x = scaler.transform(test_x)		#apply the same transformation to the validation vectors.
										#You will need to save this scaler object using pickle if you want
										#to apply it to the test dataset as well

print train_x.shape

clf = LogisticRegression(
							penalty = "l2",				#twiddle these knobs as per sklearn docs on linear models.
							C = 0.1,					#or experiment with different classifiers
							solver = "lbfgs",
							n_jobs = -1
						)

t = time()
clf.fit(train_x, train_y)								#Train the model
print "Done training, took %f seconds" % (time() - t)

thresh = 0.5

probs = clf.predict_proba(test_x)[:,1]					#Obtain probabilites that the email is opened
test_pred = [int(x>=thresh) for x in probs]				#Filter according to the threshold

acc = accuracy_score(test_y, test_pred)					#evaluate performance using various metrics.
pre = precision_score(test_y, test_pred)				#there is a good explanation of all these things on the sklearn website
rec = recall_score(test_y, test_pred)
fsc = f1_score(test_y, test_pred)
auc = roc_auc_score(test_y, probs)

print "Accuracy :", acc
print "F-Score :", fsc
print "Precision :", pre
print "Recall :", rec
print "AUC :", auc