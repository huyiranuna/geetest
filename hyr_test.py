# -*- coding: utf-8 -*-
import datetime
import pandas as pd
import os
import json
import numpy as np
import scipy.stats
import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from scipy.stats import randint as sp_randint
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from  sklearn.cluster import KMeans
from sklearn.neighbors import LSHForest
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

# return all file paths in a path
def get_file_list(fpath):
    file_path_list = []
    list1 = os.listdir(fpath)
    for i in list1:
        file_path = fpath + '\\' + i
        if os.path.isfile(file_path):
            file_path_list.append(file_path)
        else:
            list2 = get_file_list(file_path)
            file_path_list = file_path_list + list2
    return file_path_list

# return data in a file
def get_file_data(file_path):
    file_data = []
    file = open(file_path)
    for line in file.readlines():
        d = json.loads(line)
        file_data.append(d)
    return file_data

# judge whether a file is needed
def judge_file(fpath):
    if os.path.getsize(fpath) == 0:
        return False
    if os.path.splitext(fpath)[1] == '.crc':
        return False
    return True

# return needed data in a path
def get_data(fpath):
    data_df = pd.DataFrame(columns=list(range(16)))
    file_path = get_file_list(fpath)
    for i in file_path:
        if judge_file(i):
            data = get_file_data(i)
            df = pd.DataFrame(data)
            data_df = data_df.append(df)
    return data_df


def KL_work(Xs_train):
# 训练集散度
    t_people = np.array(Xs_train)[:90000]
    t_robot = np.array(Xs_train)[90001:]
    KL = np.zeros(16)
    plt.clf()
    plt.figure(figsize=(16, 16))
    for i in range(16):
        a = np.histogram(t_people[:, i], bins=50)
        b = np.histogram(t_robot[:, i], bins=a[1])
        peo = a[0] / a[0].sum()
        rob = b[0] / b[0].sum() + 1e-12
        KL[i] = scipy.stats.entropy(peo, rob)
        plt.subplot(4, 4, i + 1)
        plt.title("KL%d: %.3f" % (i,KL[i]))
        plt.hist(t_people[:, i], a[1], color='green', alpha=0.5)
        plt.hist(t_robot[:, i], a[1], color='red', alpha=0.5)
    plt.show()

def KL_work2(Xs_test):
# 测试集散度
    t_people = np.array(Xs_test)[:43955]
    t_robot = np.array(Xs_test)[43956:]
    KL = np.zeros(16)
    plt.clf()
    plt.figure(figsize=(16, 16))
    for i in range(16):
        a = np.histogram(t_people[:, i], bins=50)
        b = np.histogram(t_robot[:, i], bins=a[1])
        peo = a[0] / a[0].sum()
        rob = b[0] / b[0].sum() + 1e-12
        KL[i] = scipy.stats.entropy(peo, rob)
        plt.subplot(4, 4, i + 1)
        plt.title("KL%d: %.3f" % (i,KL[i]))
        plt.hist(t_people[:, i], a[1], color='green', alpha=0.5)
        plt.hist(t_robot[:, i], a[1], color='red', alpha=0.5)
    plt.show()

def LR_work(Xs_train, Xs_test, Ys_train, Ys_test):
# LR + RandomizedSearchCV
    pipe_LR = Pipeline([('sc', StandardScaler()), ('PCA', PCA(n_components=2, copy='true')), ('clf', LogisticRegression())])
    LR_param_dist = {'clf__tol':scipy.stats.expon(scale=.1),
                     'clf__fit_intercept':[True,False],
                     'clf__class_weight':['balanced', None],
                     "clf__C": scipy.stats.expon(scale=100),
                     "clf__solver": ['liblinear', 'lbfgs', 'newton-cg','sag']}
    search = RandomizedSearchCV(pipe_LR, LR_param_dist)
    search.fit(Xs_train, Ys_train)
    print("LR: ")
    my_confusion_matrix(Ys_test, search.predict(Xs_test))
    my_classification_report(Ys_test, search.predict(Xs_test))
    print(search.best_estimator_)


def SVM_work(Xs_train, Xs_test, Ys_train, Ys_test):
# SVM + RandomizedSearchCV
    pipe_SVC = Pipeline([('sc', StandardScaler()), ('PCA', PCA(n_components=2, copy='true')), ('clf', SVC(random_state=1))])
    SVC_param_dist = {'clf__tol': scipy.stats.expon(scale=.1),
                      # 'clf__gamma': scipy.stats.expon(scale=.1),
                      "clf__C": scipy.stats.expon(scale=100),
                      "clf__kernel": ['linear', 'rbf']}
    search = RandomizedSearchCV(pipe_SVC,param_distributions=SVC_param_dist)
    search.fit(Xs_train, Ys_train)
    print("SVC: ")
    my_confusion_matrix(Ys_test,search.predict(Xs_test))
    my_classification_report(Ys_test,search.predict(Xs_test))

def RF_work(Xs_train, Xs_test, Ys_train, Ys_test):
# RF + RandomizedSearchCV
    RF_param_dist = { "max_depth": sp_randint(1, 20),
                      "max_features": sp_randint(1, 16),
                      "min_samples_split": sp_randint(2, 50),
                      "min_samples_leaf": sp_randint(1, 50),
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}
    search = RandomizedSearchCV(RandomForestClassifier(), RF_param_dist)
    search.fit(Xs_train, Ys_train)
    print("RF: ")
    my_confusion_matrix(Ys_test, search.predict(Xs_test))
    my_classification_report(Ys_test, search.predict(Xs_test))
    print(search.best_estimator_)



def LSH_work(Xs_train, Xs_test, Ys_train, Ys_test):
# LSH + KNN
    search = LSHForest(random_state=42)
    search.fit(Xs_train)
    distances, neighbors = search.kneighbors(Xs_test, n_neighbors=2)

    y_pred = np.zeros(len(Ys_test))

    for i in range(Xs_test.shape[0]):
        n_list = neighbors[i].tolist()
        ny_list = [Ys_train[j] for j in n_list]
        y_pred[i] = 0 if ny_list.count(0)>ny_list.count(1) else 1

    print("LSH: ")
    my_confusion_matrix(Ys_test, y_pred)
    my_classification_report(Ys_test, y_pred)


def KNN(Xs_train, Xs_test, Ys_train, Ys_test):
# KNN + RandomizedSearchCV
    pipe_KNN = Pipeline([('sc', StandardScaler()), ('PCA', PCA(n_components=2, copy='true')), ('clf', KNeighborsClassifier())])
    KNN_param_dist = {"clf__n_neighbors": sp_randint(5, 20),
                      "clf__algorithm": ['ball_tree', 'kd_tree'],
                      "clf__weights": ['uniform', 'distance']
                      }
    search = RandomizedSearchCV(pipe_KNN, KNN_param_dist)
    search.fit(Xs_train,Ys_train)
    print("KNN: ")
    my_confusion_matrix(Ys_test,search.predict(Xs_test))
    my_classification_report(Ys_test,search.predict(Xs_test))
    print(search.best_estimator_)


def NB_work(Xs_train, Xs_test, Ys_train, Ys_test):
# 朴素贝叶斯
    search = GaussianNB()
    search.fit(Xs_train, Ys_train)
    print("NB: ")
    my_confusion_matrix(Ys_test,search.predict(Xs_test))
    my_classification_report(Ys_test,search.predict(Xs_test))


def GBDT_work(Xs_train, Xs_test, Ys_train, Ys_test):
# GBDT + RandomizedSearchCV
    GBDT_param_dist = {"max_depth": sp_randint(3, 15),
                      "max_features": sp_randint(1, 16),
                      "min_samples_split": sp_randint(2, 50),
                      "min_samples_leaf": sp_randint(1, 50)}
    search = RandomizedSearchCV(GradientBoostingClassifier(), GBDT_param_dist)
    search.fit(Xs_train, Ys_train)
    print("GBDT: ")
    my_confusion_matrix(Ys_test,search.predict(Xs_test))
    my_classification_report(Ys_test,search.predict(Xs_test))
    print(search.best_estimator_)


def Vote_work(Xs_train, Xs_test, Ys_train, Ys_test):
# 模型组合
    starttime = datetime.datetime.now()
    model1 = GaussianNB()
    model2 = DecisionTreeClassifier()
    search = VotingClassifier(estimators=[('NB', model1), ('DT', model2)])
    search.fit(Xs_train, Ys_train)
    print("Vote(NB,DT): ")
    my_confusion_matrix(Ys_test,search.predict(Xs_test))
    my_classification_report(Ys_test,search.predict(Xs_test))
    endtime = datetime.datetime.now()
    print("run time: ", (endtime - starttime).seconds, "s")


def KMeans_work(Xs_train,Ys_train,Xs_test,Ys_test):
# Kmeans + 分类模型
    Ys_train_2 = np.zeros(180000)  #训练数据新标签
    search = KMeans(n_clusters=4).fit(Xs_train)
    print(search.labels_)
    labels = search.labels_.tolist()

    for i in range(len(labels)):
    #根据聚类结果和数据标签组合为训练数据的新标签
        if labels[i] == 0 and i < 90000:
            Ys_train_2[i] = 0
        elif labels[i] == 1 and i < 90000:
            Ys_train_2[i] = 1
        elif labels[i] == 2 and i < 90000:
            Ys_train_2[i] = 2
        elif labels[i] == 3 and i < 90000:
            Ys_train_2[i] = 3
        elif labels[i] == 0 and i >= 90000:
            Ys_train_2[i] = 4
        elif labels[i] == 1 and i >= 90000:
            Ys_train_2[i] = 5
        elif labels[i] == 2 and i >= 90000:
            Ys_train_2[i] = 6
        else:
            Ys_train_2[i] = 7

    search1 = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini',min_samples_leaf=15, min_samples_split=8),
                                 algorithm='SAMME',
                                 learning_rate=0.8,
                                 n_estimators=185 )
    search1.fit(Xs_train, Ys_train_2)

    y_pred1 = search1.predict(Xs_test).tolist()  #测试数据的预测分类（新标签）
    y_pred = np.zeros(len(y_pred1))
    for i in range(len(y_pred1)):
    #根据预测结果将新标签转化为老标签
        if y_pred1[i] == 0 or y_pred1[i] == 1 or y_pred1[i] == 2 or y_pred1[i] == 3:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    print("KMeans&AdaBoost: ")
    my_confusion_matrix(Ys_test, y_pred)
    my_classification_report(Ys_test, y_pred)



def KMeans_work_2(Xs_train,Ys_train,Xs_test,Ys_test):
# KMeans预测
    starttime = datetime.datetime.now()
    search = KMeans(n_clusters=50).fit(Xs_train)
    labels = search.labels_.tolist()
    labels_y = np.zeros(50)

    for e in range(50):
        e_index = [i for i, x in enumerate(labels) if x == e]
        y = [Ys_train[j] for j in e_index]
        labels_y[e] = 0 if y.count(0) > y.count(1) else 1

    y_p = search.predict(Xs_test).tolist()
    y_pred = [labels_y[x] for x in y_p]

    print("KMeans(50): ")
    my_confusion_matrix(Ys_test, y_pred)
    my_classification_report(Ys_test, y_pred)
    endtime = datetime.datetime.now()
    print("run time: ", (endtime - starttime).seconds, "s")

def my_confusion_matrix(y_true, y_pred):
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    print("confusion_matrix(left labels: y_true, up labels: y_pred):")
    print("\t",end='')
    for i in range(len(labels)):
        print(labels[i],"\t",end='')
    print()
    for i in range(len(conf_mat)):
        print(i,"\t",end='')
        for j in range(len(conf_mat[i])):
            print(conf_mat[i][j],'\t',end='')
        print()
    print()

def my_classification_report(y_true, y_pred):
    print("classification_report(left: labels):")
    print(classification_report(y_true, y_pred,digits=4))
    print(accuracy_score(y_true, y_pred))


if __name__ == '__main__':
    print(datetime.datetime.now())
    train_people = get_data('D:\\PycharmProjects\\untitled\\gt-kaggle\\train_data\\train_people')
    train_robot = get_data('D:\\PycharmProjects\\untitled\\gt-kaggle\\train_data\\train_robot')
    test_people = get_data('D:\\PycharmProjects\\untitled\\gt-kaggle\\test_data\\test_people')
    test_robot_c = get_data('D:\\PycharmProjects\\untitled\\gt-kaggle\\test_data\\test_robot_c')
    test_robot_Cl = get_data('D:\\PycharmProjects\\untitled\\gt-kaggle\\test_data\\test_robot_Cl')

    Xs_train = train_people.sample(n=90000).append(train_robot.sample(n=90000))
    Ys_train = [0]*90000 + [1]*90000

    Xs_test = test_people.append(test_robot_c).append(test_robot_Cl)
    Ys_test = [0]*test_people.shape[0] + [1]*(test_robot_Cl.shape[0]+test_robot_c.shape[0])


    # KL_work(Xs_train)
    # KL_work2(Xs_test)
    # KMeans_work(Xs_train,Ys_train,Xs_test,Ys_test)
    # Xs_train, Xs_test, Ys_train, Ys_test = train_test_split(train_data, train_flag, test_size=0.20, random_state=1)
    # LR_work(Xs_train, Xs_test, Ys_train, Ys_test)
    # RF_work(Xs_train, Xs_test, Ys_train, Ys_test)
    # NB_work(Xs_train, Xs_test, Ys_train, Ys_test)
    # SVM_work(Xs_train, Xs_test, Ys_train, Ys_test)
    # GBDT_work(Xs_train, Xs_test, Ys_train, Ys_test)
    # LSH_work(Xs_train, Xs_test, Ys_train, Ys_test)
    # KMeans_work(Xs_train, Ys_train, Xs_test, Ys_test)
    print(datetime.datetime.now())




