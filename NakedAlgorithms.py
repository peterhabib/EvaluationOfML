# Libraries
import datetime

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, homogeneity_score, adjusted_rand_score, \
    roc_auc_score, roc_curve, f1_score, auc
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
from sklearn.svm import NuSVC
import time
from sklearn.svm.classes import OneClassSVM
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve



#To ignore warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

EvalFile = open('EvaluationFile(50%).csv','w')

data = pd.read_csv('/home/peter/Desktop/MamogAI/Scripts/breastCancer.csv')
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

M = data[data.diagnosis=='M']
B = data[data.diagnosis=='B']
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

y=data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

x = x_data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=50)


algorithms = ['KNeighborsClassifier','GaussianNB','SGDClassifier','ExtraTreeClassifier','DecisionTreeClassifier',
              'MLPClassifier','RidgeClassifierCV','RidgeClassifier','GaussianProcessClassifier',
              'AdaBoostClassifier','GradientBoostingClassifier','BaggingClassifier','ExtraTreesClassifier','RandomForestClassifier',
              'CalibratedClassifierCV','LinearDiscriminantAnalysis','LinearSVC',
              'LogisticRegression','LogisticRegressionCV','NearestCentroid','Perceptron','QuadraticDiscriminantAnalysis',
              'MultinomialNB']
EvalFile.write('algorithm\talgorithm_score\tmean_absolute_error\tmean_squared_error\tr2_score\t'
               'homogeneity_score\tadjusted_rand_score\tv_measure_score\tTimeToBuild\tNo Skill\tF-Measure\tAUC\tCrossValidation\n')
for algo in algorithms:
    start_time = time.time()
    model = eval('%s()'%algo)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    EvalFile
    end = time.time() - start_time



    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = y_pred
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc1 = roc_auc_score(y_test, lr_probs)
    # summarize scores

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='%s'%algo)
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    # pyplot.show()
    # pyplot.savefig("%s.svg"%algo)

    yhat = y_pred
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
    lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = len(y[y == 1]) / len(y)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(lr_recall, lr_precision, marker='.', label='%s'%algo)
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    # pyplot.show()
    # pyplot.savefig("%s.svg" % algo)
    # print(cross_val_score(model, x_train, y_train, cv=100) * 100)
    print("Accuracy of %s algorithm : "%algo, model.score(x_test, y_test) * 100)
    print("Accuracy of mean_absolute_error algorithm : ", mean_absolute_error(y_test, y_pred) * 100)
    print("Accuracy of mean_squared_error algorithm : ", mean_squared_error(y_test, y_pred) * 100)
    print("Accuracy of r2_score algorithm : ", r2_score(y_test, y_pred) * 100)
    print("Accuracy of homogeneity_score algorithm : ", homogeneity_score(y_test, y_pred) * 100)
    print("Accuracy of adjusted_rand_score algorithm : ", adjusted_rand_score(y_test, y_pred) * 100)
    print("Accuracy of v_measure_score algorithm : ", metrics.v_measure_score(y_test, y_pred) * 100)
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('%s: ROC AUC=%.3f' % (algo, lr_auc1))
    # print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    print('----------------------------------------------------')
    print('----------------------------------------------------')

    ToWrite='%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (algo,
                                                                  round(model.score(x_test, y_test) * 100,1),
                                                                  round(mean_absolute_error(y_test, y_pred) * 100,1),
                                                                  round( mean_squared_error(y_test, y_pred) * 100,1),
                                                                  round(r2_score(y_test, y_pred) * 100,1),
                                                                  round(homogeneity_score(y_test, y_pred) * 100,1),
                                                                  round(adjusted_rand_score(y_test, y_pred) * 100,1),
                                                                  round(metrics.v_measure_score(y_test, y_pred) * 100,1),
                                                                  round(end, 2),
                                                                  round(ns_auc,2),
                                                                  round(lr_auc1,2),
                                                                  round(lr_f1,2))

                                                                  # str(cross_val_score(model, x_train, y_train,cv=10) * 100))

    EvalFile.write(ToWrite)
EvalFile.close()


