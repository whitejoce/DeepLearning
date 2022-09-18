#!/usr/bin/env python
# _*_coding: utf-8 _*_
#Coder:Whitejoce

from random import random

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_data=datasets.load_iris()
'''
for i in range(0,len(iris_data["data"])):
    print(iris_data["target_names"][iris_data["target"][i]],end=",")
    for j in range(0,len(iris_data["data"][i])):
        print(iris_data["data"][i][j],end=" ")
    print()
'''
X_train,X_test,Y_train,Y_test=train_test_split(iris_data["data"],iris_data["target"],random_state=1)

knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,Y_train)
Y_pred=knn.predict(X_test)

print(" [+] KNN模型精度: {:.2f}%".format(knn.score(X_test,Y_test)*100))
for i in range(0,5):
    random_num=int(random()*150)
    test_data=np.array([iris_data["data"][random_num]],dtype=np.float32)
    ans=iris_data["target_names"][iris_data["target"][random_num]]
    print(" [=] 预测数据: {}".format(test_data))
    if iris_data['target_names'][knn.predict(test_data)][0]==ans:
        print(" [+] 预测结果(正确): {} ".format(iris_data['target_names'][knn.predict(test_data)][0]))
    else:
        print(" [-] 预测结果(错误): {} ".format(iris_data['target_names'][knn.predict(test_data)][0]))
