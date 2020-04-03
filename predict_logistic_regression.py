#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 00:19:20 2020

@author: Arijeet Singh
"""
import pandas as pd
import utils
from sklearn import linear_model, preprocessing

train = pd.read_csv("train.csv")
utils.clean_data(train)

target = train['Survived'].values
features_names = ['Pclass','Age','Sex','SibSp','Parch','Fare', 'Embarked']
features = train[features_names].values

classifier = linear_model.LogisticRegression()

classifier_ = classifier.fit(features, target)
print(classifier_.score(features,target))

poly = preprocessing.PolynomialFeatures(degree = 2)
poly_features = poly.fit_transform(features)

classifier_ = classifier.fit(poly_features, target)
print(classifier_.score(poly_features,target))