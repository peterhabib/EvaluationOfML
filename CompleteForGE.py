# Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import sys

from sklearn.ensemble import RandomForestClassifier

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


data = pd.read_csv('breastCancer.csv')

data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
# print(data.tail())


M = data[data.diagnosis=='M']
B = data[data.diagnosis=='B']
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})



y=data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)




x = x_data
from sklearn.model_selection import train_test_split, cross_val_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=50)

# from sklearn import neighbors, datasets, preprocessing
# X, y = data[:], data.diagnosis
# x_train, x_test, y_train, y_test = train_test_split(X, y)
# scaler = preprocessing.StandardScaler().fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
#
# from sklearn.preprocessing import Normalizer
# scaler = Normalizer().fit(x_train)
# normalized_X = scaler.transform(x_train)
# normalized_X_test = scaler.transform(x_test)

# from sklearn.preprocessing import Imputer
# imp = Imputer(missing_values=0, strategy='mean', axis=0)
# x_train = imp.fit_transform(x_train)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, homogeneity_score, adjusted_rand_score, \
    f1_score, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import v_measure_score
from sklearn import metrics
# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve

knn = artificial_neural_network()
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print("Accuracy of KNN algorithm : ",knn.score(x_test,y_test)*100)
print("Accuracy of mean_absolute_error algorithm : ",mean_absolute_error(y_test, y_pred)*100)
print("Accuracy of mean_squared_error algorithm : ",mean_squared_error(y_test, y_pred)*100)
print("Accuracy of r2_score algorithm : ",r2_score(y_test, y_pred)*100)
print("Accuracy of homogeneity_score algorithm : ",homogeneity_score(y_test, y_pred)*100)
print("Accuracy of adjusted_rand_score algorithm : ",adjusted_rand_score(y_test, y_pred)*100)
print("Accuracy of v_measure_score algorithm : ",metrics.v_measure_score(y_test, y_pred)*100)

print(cross_val_score(knn, x_train, y_train, cv=100)*100)





ns_probs = [0 for _ in range(len(y_test))]
lr_probs = y_pred
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()



yhat = y_pred
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y[y==1]) / len(y)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# from sklearn.svm import SVC
# svm = SVC()
# svm.fit(x_train,y_train)
# y_pred = svm.predict(x_test)
# print("Accuracy of SVM algorithm : ",svm.score(x_test,y_pred)*100)
# print("Accuracy of mean_absolute_error algorithm : ",mean_absolute_error(y_test, y_pred)*100)
# print("Accuracy of mean_squared_error algorithm : ",mean_squared_error(y_test, y_pred)*100)
# print("Accuracy of r2_score algorithm : ",r2_score(y_test, y_pred)*100)
# print("Accuracy of homogeneity_score algorithm : ",homogeneity_score(y_test, y_pred)*100)
# print("Accuracy of adjusted_rand_score algorithm : ",adjusted_rand_score(y_test, y_pred)*100)
#
#
#
#
# # Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# Naive = GaussianNB()
# Naive.fit(x_train,y_train)
# y_pred = Naive.predict(x_test)
# print("Accuracy of GaussianNB algorithm : ",Naive.score(x_test,y_pred)*100)
# print("Accuracy of mean_absolute_error algorithm : ",mean_absolute_error(y_test, y_pred)*100)
# print("Accuracy of mean_squared_error algorithm : ",mean_squared_error(y_test, y_pred)*100)
# print("Accuracy of r2_score algorithm : ",r2_score(y_test, y_pred)*100)
# print("Accuracy of homogeneity_score algorithm : ",homogeneity_score(y_test, y_pred)*100)
# print("Accuracy of adjusted_rand_score algorithm : ",adjusted_rand_score(y_test, y_pred)*100)
#
#
#
# # SGDClassifier
# from sklearn.linear_model import SGDClassifier
# SGDCl = SGDClassifier()
# SGDCl.fit(x_train,y_train)
# y_pred = SGDCl.predict(x_test)
# print("Accuracy of SGDClassifier algorithm : ",SGDCl.score(x_test,y_pred)*100)
# print("Accuracy of mean_absolute_error algorithm : ",mean_absolute_error(y_test, y_pred)*100)
# print("Accuracy of mean_squared_error algorithm : ",mean_squared_error(y_test, y_pred)*100)
# print("Accuracy of r2_score algorithm : ",r2_score(y_test, y_pred)*100)
# print("Accuracy of homogeneity_score algorithm : ",homogeneity_score(y_test, y_pred)*100)
# print("Accuracy of adjusted_rand_score algorithm : ",adjusted_rand_score(y_test, y_pred)*100)
#
#
#
#
# # SGDRegressor
# from sklearn.linear_model import SGDRegressor
# SGDRe = SGDRegressor()
# SGDRe.fit(x_train,y_train)
# y_pred = SGDRe.predict(x_test)
# print("Accuracy of SGDRegressor algorithm : ",SGDRe.score(x_test,y_pred)*100)
# print("Accuracy of mean_absolute_error algorithm : ",mean_absolute_error(y_test, y_pred)*100)
# print("Accuracy of mean_squared_error algorithm : ",mean_squared_error(y_test, y_pred)*100)
# print("Accuracy of r2_score algorithm : ",r2_score(y_test, y_pred)*100)
# print("Accuracy of homogeneity_score algorithm : ",homogeneity_score(y_test, y_pred)*100)
# print("Accuracy of adjusted_rand_score algorithm : ",adjusted_rand_score(y_test, y_pred)*100)
#
#
#
# #KMeans
# from sklearn.cluster import KMeans
# k_means = KMeans()
# k_means.fit(x_train,y_train)
# print("Accuracy of KMeans Algorithm",k_means.score(x_test,y_test)*100)
#
#
# #ExtraTreeClassifier
# from sklearn.tree import ExtraTreeClassifier
# tree = ExtraTreeClassifier()
# tree.fit(x_train,y_train)
# print("Accuracy of ExtraTreeClassifier Algorithm",tree.score(x_test,y_test)*100)
#
#
# #DecisionTreeClassifier
# from sklearn.tree import DecisionTreeClassifier
# tree = DecisionTreeClassifier()
# tree.fit(x_train,y_train)
# print("Accuracy of DecisionTreeClassifier Algorithm",tree.score(x_test,y_test)*100)
#
#
#
#
# # OneClassSVM Error
# from sklearn.svm.classes import OneClassSVM
# svmclasses = OneClassSVM()
# svmclasses.fit(x_train,y_train)
# print("Accuracy of DecisionTreeClassifier Algorithm",svmclasses.score_samples(x_test)*100)
#
#
# # MLPClassifier
# from sklearn.neural_network.multilayer_perceptron import MLPClassifier
# neural_network = MLPClassifier(max_iter=500, early_stopping=False)
# neural_network.fit(x_train,y_train)
# print("Accuracy of MLPClassifier Algorithm",neural_network.score(x_test,y_test)*100)
#
#
# #RadiusNeighborsClassifier
# from sklearn.neighbors.classification import RadiusNeighborsClassifier
# neighbors = DecisionTreeClassifier()
# neighbors.fit(x_train,y_train)
# print("Accuracy of RadiusNeighborsClassifier Algorithm",neighbors.score(x_test,y_test)*100)
#
#
#
# # ClassifierChain
# # from sklearn.multioutput import ClassifierChain
# # multioutput = ClassifierChain(base_estimator='SGDClassifier')
# # multioutput.fit(x_train,y_train)
# # print("Accuracy of ClassifierChain Algorithm",multioutput.score(x_test,y_test)*100)
#
#
#
# # MultiOutputClassifier Error
# # from sklearn.multioutput import MultiOutputClassifier
# # multioutput = MultiOutputClassifier(estimator='predict_proba')
# # multioutput.fit(x_train,y_train)
# # print("Accuracy of ClassifierChain Algorithm",multioutput.score(x_test,y_test)*100)
#
#
# #OutputCodeClassifier Error
# # from sklearn.multiclass import OutputCodeClassifier
# # multiclass = OutputCodeClassifier(estimator='decision_function')
# # multiclass.fit(x_train,y_train)
# # print("Accuracy of ClassifierChain Algorithm",multiclass.score(x_test,y_test)*100)
#
# #OneVsOneClassifier Error
# # from sklearn.multiclass import OneVsOneClassifier
# # multiclass = OneVsOneClassifier()
# # multiclass.fit(x_train,y_train)
# # print("Accuracy of ClassifierChain Algorithm",multiclass.score(x_test,y_test)*100)
#
# #RidgeClassifierCV Error
# from sklearn.linear_model.ridge import RidgeClassifierCV
# ridge = RidgeClassifierCV()
# ridge.fit(x_train,y_train)
# print("Accuracy of RidgeClassifierCV Algorithm",ridge.score(x_test,y_test)*100)
#
# #RidgeClassifier Error
# from sklearn.linear_model.ridge import RidgeClassifier
# ridge = RidgeClassifier()
# ridge.fit(x_train,y_train)
# print("Accuracy of RidgeClassifier Algorithm",ridge.score(x_test,y_test)*100)
#
# #PassiveAggressiveClassifier Error
# from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
# passive_aggressive = PassiveAggressiveClassifier()
# passive_aggressive.fit(x_train,y_train)
# print("Accuracy of PassiveAggressiveClassifier Algorithm",passive_aggressive.score(x_test,y_test)*100)
#
# #GaussianProcessClassifier Error
# from sklearn.gaussian_process.gpc import GaussianProcessClassifier
# gpc = GaussianProcessClassifier()
# gpc.fit(x_train,y_train)
# print("Accuracy of GaussianProcessClassifier Algorithm",gpc.score(x_test,y_test)*100)
#
# #GaussianProcessClassifier
# # from sklearn.ensemble.voting import VotingClassifier
# # voting = VotingClassifier()
# # voting.fit(x_train,y_train)
# # print("Accuracy of GaussianProcessClassifier Algorithm",voting.score(x_test,y_test)*100)
#
#
# #AdaBoostClassifier
# from sklearn.ensemble.weight_boosting import AdaBoostClassifier
# weight_boosting = AdaBoostClassifier()
# weight_boosting.fit(x_train,y_train)
# print("Accuracy of AdaBoostClassifier Algorithm",weight_boosting.score(x_test,y_test)*100)
#
#
# #GradientBoostingClassifier
# from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
# gradient_boosting = GradientBoostingClassifier()
# gradient_boosting.fit(x_train,y_train)
# print("Accuracy of GradientBoostingClassifier Algorithm",gradient_boosting.score(x_test,y_test)*100)
#
#
# #BaggingClassifier
# from sklearn.ensemble.bagging import BaggingClassifier
# bagging = BaggingClassifier()
# bagging.fit(x_train,y_train)
# print("Accuracy of BaggingClassifier Algorithm",bagging.score(x_test,y_test)*100)
#
#
# #ExtraTreesClassifier
# from sklearn.ensemble.forest import ExtraTreesClassifier
# forest = ExtraTreesClassifier()
# forest.fit(x_train,y_train)
# print("Accuracy of ExtraTreesClassifier Algorithm",forest.score(x_test,y_test)*100)
#
#
#
# #RandomForestClassifier
# from sklearn.ensemble.forest import RandomForestClassifier
# forest = RandomForestClassifier()
# forest.fit(x_train,y_train)
# print("Accuracy of RandomForestClassifier Algorithm",forest.score(x_test,y_test)*100)
#
#
#
# #BernoulliNB
# from sklearn.naive_bayes import BernoulliNB
# naive_bayes = BernoulliNB()
# naive_bayes.fit(x_train,y_train)
# print("Accuracy of BernoulliNB Algorithm",naive_bayes.score(x_test,y_test)*100)
#
#
# #CalibratedClassifierCV
# from sklearn.calibration import CalibratedClassifierCV
# calibration = CalibratedClassifierCV()
# calibration.fit(x_train,y_train)
# print("Accuracy of CalibratedClassifierCV Algorithm",calibration.score(x_test,y_test)*100)
#
#
# #LabelPropagation
# from sklearn.semi_supervised import LabelPropagation
# semi_supervised = LabelPropagation()
# semi_supervised.fit(x_train,y_train)
# print("Accuracy of LabelPropagation Algorithm",semi_supervised.score(x_test,y_test)*100)
#
#
# #LabelSpreading
# from sklearn.semi_supervised import LabelSpreading
# semi_supervised = LabelSpreading()
# semi_supervised.fit(x_train,y_train)
# print("Accuracy of LabelSpreading Algorithm",semi_supervised.score(x_test,y_test)*100)
#
# #LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# discriminant_analysis = LinearDiscriminantAnalysis()
# discriminant_analysis.fit(x_train,y_train)
# print("Accuracy of LinearDiscriminantAnalysis Algorithm",discriminant_analysis.score(x_test,y_test)*100)
# print(discriminant_analysis.predict(x_test))
# print(y_test)
#
# #LinearDiscriminantAnalysis
# from sklearn.svm import LinearSVC
# svm = LinearSVC()
# svm.fit(x_train,y_train)
# print("Accuracy of LinearSVC Algorithm",svm.score(x_test,y_test)*100)
#
#
# #LogisticRegression
# from sklearn.linear_model import LogisticRegression
# linear_model = LogisticRegression()
# linear_model.fit(x_train,y_train)
# print("Accuracy of LogisticRegression Algorithm",linear_model.score(x_test,y_test)*100)
#
#
#
# #LogisticRegressionCV
# from sklearn.linear_model import LogisticRegressionCV
# linear_modelcv = LogisticRegressionCV()
# linear_modelcv.fit(x_train,y_train)
# print("Accuracy of LogisticRegressionCV Algorithm",linear_modelcv.score(x_test,y_test)*100)
#
#
#
# #MultinomialNB
# # from sklearn.naive_bayes import MultinomialNB
# # naive_bayes = MultinomialNB()
# # naive_bayes.fit(x_train,y_train)
# # print("Accuracy of MultinomialNB Algorithm",naive_bayes.score(x_test,y_test)*100)
# #
#
#
# #MultinomialNB
# from sklearn.neighbors import NearestCentroid
# neighbors = NearestCentroid()
# neighbors.fit(x_train,y_train)
# print("Accuracy of MultinomialNB Algorithm",neighbors.score(x_test,y_test)*100)
#
#
#
# #NuSVC
# from sklearn.svm import NuSVC
# svm = NuSVC()
# svm.fit(x_train,y_train)
# print("Accuracy of NuSVC Algorithm",svm.score(x_test,y_test)*100)
#
#
#
# #Perceptron
# from sklearn.linear_model import Perceptron
# linear_model = Perceptron()
# linear_model.fit(x_train,y_train)
# print("Accuracy of Perceptron Algorithm",linear_model.score(x_test,y_test)*100)
#
# #Perceptron
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# discriminant_analysis = QuadraticDiscriminantAnalysis()
# discriminant_analysis.fit(x_train,y_train)
# print("Accuracy of QuadraticDiscriminantAnalysis Algorithm",discriminant_analysis.score(x_test,y_test)*100)
#
#
# #GMM
# from sklearn.mixture import GaussianMixture
# mixture = GaussianMixture()
# mixture.fit(x_train,y_train)
# print("Accuracy of GaussianMixture Algorithm",mixture.score(x_test,y_test)*100)

