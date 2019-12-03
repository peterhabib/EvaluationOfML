# Libraries
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

EvalFile = open('EvaluationFile(Scaled).csv','w')

data = pd.read_csv('breastCancer.csv')
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

M = data[data.diagnosis=='M']
B = data[data.diagnosis=='B']
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

y=data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

x = x_data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=50)



from sklearn import neighbors, datasets, preprocessing
X, y = data[:], data.diagnosis
x_train, x_test, y_train, y_test = train_test_split(X, y)
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


algorithms = ['KNeighborsClassifier','SVC','GaussianNB','SGDClassifier','SGDRegressor','KMeans','ExtraTreeClassifier','DecisionTreeClassifier',
              'MLPClassifier','RidgeClassifierCV','RidgeClassifier','PassiveAggressiveClassifier','GaussianProcessClassifier',
              'AdaBoostClassifier','GradientBoostingClassifier','BaggingClassifier','ExtraTreesClassifier','RandomForestClassifier',
              'BernoulliNB','CalibratedClassifierCV','LabelPropagation','LabelSpreading','LinearDiscriminantAnalysis','LinearSVC',
              'LogisticRegression','LogisticRegressionCV','NearestCentroid','Perceptron','QuadraticDiscriminantAnalysis',
              'GaussianMixture']

EvalFile.write('algorithm\talgorithm_score\tmean_absolute_error\tmean_squared_error\tr2_score\thomogeneity_score\tadjusted_rand_score\tv_measure_score\n')
for algo in algorithms:
    model = eval('%s()'%algo)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    EvalFile
    print("Accuracy of %s algorithm : "%algo, model.score(x_test, y_test) * 100)
    print("Accuracy of mean_absolute_error algorithm : ", mean_absolute_error(y_test, y_pred) * 100)
    print("Accuracy of mean_squared_error algorithm : ", mean_squared_error(y_test, y_pred) * 100)
    print("Accuracy of r2_score algorithm : ", r2_score(y_test, y_pred) * 100)
    print("Accuracy of homogeneity_score algorithm : ", homogeneity_score(y_test, y_pred) * 100)
    print("Accuracy of adjusted_rand_score algorithm : ", adjusted_rand_score(y_test, y_pred) * 100)
    print("Accuracy of v_measure_score algorithm : ", metrics.v_measure_score(y_test, y_pred) * 100)
    print('----------------------------------------------------')
    print('----------------------------------------------------')
    ToWrite='%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (algo,
                                                  round(model.score(x_test, y_test) * 100),
                                                  round(mean_absolute_error(y_test, y_pred) * 100),
                                                  round( mean_squared_error(y_test, y_pred) * 100),
                                                  round(r2_score(y_test, y_pred) * 100),
                                                  round(homogeneity_score(y_test, y_pred) * 100),
                                                  round(adjusted_rand_score(y_test, y_pred) * 100),
                                                  round(metrics.v_measure_score(y_test, y_pred) * 100))

    EvalFile.write(ToWrite)
EvalFile.close()


