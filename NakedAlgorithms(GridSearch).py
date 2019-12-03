# Libraries
from sklearn.model_selection import learning_curve, GridSearchCV
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,homogeneity_score,adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import v_measure_score
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,homogeneity_score,adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import v_measure_score
from sklearn import metrics
import time


#To ignore warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# EvalFile = open('EvaluationFile.csv','w')

data = pd.read_csv('/home/peter/Desktop/MamogAI/breastCancer.csv')
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

M = data[data.diagnosis=='M']
B = data[data.diagnosis=='B']
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

y=data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

x = x_data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=50)



# knn = KNeighborsClassifier()
# knn.fit(x_train,y_train)
# y_pred = knn.predict(x_test)
# print("Accuracy of KNN algorithm : ",knn.score(x_test,y_test)*100)
# print("Accuracy of mean_absolute_error algorithm : ",mean_absolute_error(y_test, y_pred)*100)
# print("Accuracy of mean_squared_error algorithm : ",mean_squared_error(y_test, y_pred)*100)
# print("Accuracy of r2_score algorithm : ",r2_score(y_test, y_pred)*100)
# print("Accuracy of homogeneity_score algorithm : ",homogeneity_score(y_test, y_pred)*100)
# print("Accuracy of adjusted_rand_score algorithm : ",adjusted_rand_score(y_test, y_pred)*100)
# print("Accuracy of v_measure_score algorithm : ",metrics.v_measure_score(y_test, y_pred)*100)
# print('------------------------------------------')
# params = {"n_neighbors": np.arange(1,100)}
# grid = GridSearchCV(estimator=knn,param_grid=params)
# grid.fit(x_train, y_train)
# # print(grid.best_score_)
# print(grid.best_estimator_)
# kn = grid.best_estimator_
# kn.fit(x_train,y_train)
# y_pred = kn.predict(x_test)
# print("Accuracy of KNN algorithm : ",kn.score(x_test,y_test)*100)
# print("Accuracy of mean_absolute_error algorithm : ",mean_absolute_error(y_test, y_pred)*100)
# print("Accuracy of mean_squared_error algorithm : ",mean_squared_error(y_test, y_pred)*100)
# print("Accuracy of r2_score algorithm : ",r2_score(y_test, y_pred)*100)
# print("Accuracy of homogeneity_score algorithm : ",homogeneity_score(y_test, y_pred)*100)
# print("Accuracy of adjusted_rand_score algorithm : ",adjusted_rand_score(y_test, y_pred)*100)
# print("Accuracy of v_measure_score algorithm : ",metrics.v_measure_score(y_test, y_pred)*100)


# svm = SVC()
# svm.fit(x_train,y_train)
# y_pred = svm.predict(x_test)
# print("Accuracy of SVM algorithm : ",svm.score(x_test,y_pred)*100)
# print("Accuracy of mean_absolute_error algorithm : ",mean_absolute_error(y_test, y_pred)*100)
# print("Accuracy of mean_squared_error algorithm : ",mean_squared_error(y_test, y_pred)*100)
# print("Accuracy of r2_score algorithm : ",r2_score(y_test, y_pred)*100)
# print("Accuracy of homogeneity_score algorithm : ",homogeneity_score(y_test, y_pred)*100)
# print("Accuracy of adjusted_rand_score algorithm : ",adjusted_rand_score(y_test, y_pred)*100)
# print('------------------------------------------')
#
#
# param = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]
#
#
# grid = GridSearchCV(estimator=SVC(C=1.0,
#                      verbose=False),param_grid=param)
# grid.fit(x_train, y_train)
# # print(grid.best_score_)
# print(grid.best_estimator_)
# kn = grid.best_estimator_
# kn.fit(x_train,y_train)
# y_pred = kn.predict(x_test)
# print("Accuracy of KNN algorithm : ",kn.score(x_test,y_test)*100)
# print("Accuracy of mean_absolute_error algorithm : ",mean_absolute_error(y_test, y_pred)*100)
# print("Accuracy of mean_squared_error algorithm : ",mean_squared_error(y_test, y_pred)*100)
# print("Accuracy of r2_score algorithm : ",r2_score(y_test, y_pred)*100)
# print("Accuracy of homogeneity_score algorithm : ",homogeneity_score(y_test, y_pred)*100)
# print("Accuracy of adjusted_rand_score algorithm : ",adjusted_rand_score(y_test, y_pred)*100)
# print("Accuracy of v_measure_score algorithm : ",metrics.v_measure_score(y_test, y_pred)*100)
#


from sklearn.naive_bayes import GaussianNB
Naive = GaussianNB()
Naive.fit(x_train,y_train)
y_pred = Naive.predict(x_test)
print("Accuracy of GaussianNB algorithm : ",Naive.score(x_test,y_pred)*100)
print("Accuracy of mean_absolute_error algorithm : ",mean_absolute_error(y_test, y_pred)*100)
print("Accuracy of mean_squared_error algorithm : ",mean_squared_error(y_test, y_pred)*100)
print("Accuracy of r2_score algorithm : ",r2_score(y_test, y_pred)*100)
print("Accuracy of homogeneity_score algorithm : ",homogeneity_score(y_test, y_pred)*100)
print("Accuracy of adjusted_rand_score algorithm : ",adjusted_rand_score(y_test, y_pred)*100)
print('------------------------------------------')


param = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]


grid = GridSearchCV(estimator=GaussianNB,param_grid=param)
grid.fit(x_train, y_train)
# print(grid.best_score_)
print(grid.best_estimator_)
kn = grid.best_estimator_
kn.fit(x_train,y_train)
y_pred = kn.predict(x_test)
print("Accuracy of KNN algorithm : ",kn.score(x_test,y_test)*100)
print("Accuracy of mean_absolute_error algorithm : ",mean_absolute_error(y_test, y_pred)*100)
print("Accuracy of mean_squared_error algorithm : ",mean_squared_error(y_test, y_pred)*100)
print("Accuracy of r2_score algorithm : ",r2_score(y_test, y_pred)*100)
print("Accuracy of homogeneity_score algorithm : ",homogeneity_score(y_test, y_pred)*100)
print("Accuracy of adjusted_rand_score algorithm : ",adjusted_rand_score(y_test, y_pred)*100)
print("Accuracy of v_measure_score algorithm : ",metrics.v_measure_score(y_test, y_pred)*100)