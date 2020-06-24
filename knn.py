# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 02:45:25 2020

@author: cricketjanoon
"""


import numpy as np
import pandas as pd
from scipy.stats import mode

#lables for integer indexing
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

def read_data(filename, labels):
    
    #reading file as dataframe
    df = pd.read_csv(filename)

    #storing all 4 features in numpy array
    x = np.zeros(shape=(df.shape[0], df.shape[1]-1))
    x = df[['x1', 'x2', 'x3', 'x4']].to_numpy()

    #storing y-labels as numpy array of integer values 0 for setosa, 1 for versicolor, 2 for virginica
    y = np.zeros(shape=(df.shape[0], 1))
    y = np.array([labels.index(x) for x in df['y'].to_numpy()])
    y = y[:, np.newaxis]

    return x, y

def compute_distance(x_train, x_test):
    dist = np.zeros((x_train.shape[0], x_test.shape[0]))
    for i in range(x_train.shape[0]):
        for j in range(x_test.shape[0]):
            dist[i, j] = np.sqrt(np.sum((x_train[i]-x_test[j])**2))
            
    return dist

def compute_distance_one_loop(x_train, x_test):
    dist = np.zeros((x_train.shape[0], x_test.shape[0]))
    for i in range(x_train.shape[0]):
        dist[i, :] = np.sqrt(np.sum((x_train[i, :] - x_test)**2, axis=1))
            
    return dist

def compute_distance_no_loop(x_train, x_test):
    dist = np.zeros((x_train.shape[0], x_test.shape[0]))
    dist = (x_test**2).sum(axis=1)[:, np.newaxis] + (x_train**2).sum(axis=1) - 2 * x_test.dot(x_train.T)
    dist = np.array(dist, dtype=np.float64)
    dist = np.sqrt(dist)
    
    return dist.T


def find_kNN(x_train, y_train, x_test, y_test, k):
    dist = compute_distance_one_loop(x_train, x_test)
    indices = np.argsort(dist, axis=0)

    sorted_labels = np.zeros((dist.shape[0], dist.shape[1]))
    for i in range(y_test.shape[0]):
        sorted_labels[:, i][:, np.newaxis] = np.take_along_axis(y_train, indices[:, i][:, np.newaxis], axis=0)

    k_sorted_labels = sorted_labels[:k, :]
    predicted_values, _ = mode(k_sorted_labels)
    predicted_values = predicted_values.T
    
    return predicted_values


def find_accuracy(pred, x_test):
    results = (pred == y_test)
    accuracy = np.sum(results)/ y_test.shape[0]
    return accuracy


x_train, y_train = read_data('train.csv', labels)
x_test, y_test = read_data('test.csv', labels)

k=1
pred = find_kNN(x_train, y_train, x_test, y_test, k)
acc = find_accuracy(pred, x_test)
print("For k={} accuracy is {}.".format(k, acc))

k=3
pred = find_kNN(x_train, y_train, x_test, y_test, k)
acc = find_accuracy(pred, x_test)
print("For k={} accuracy is {}.".format(k, acc))

k=5
pred = find_kNN(x_train, y_train, x_test, y_test, k)
acc = find_accuracy(pred, x_test)
print("For k={} accuracy is {}.".format(k, acc))

k=7
pred = find_kNN(x_train, y_train, x_test, y_test, k)
acc = find_accuracy(pred, x_test)
print("For k={} accuracy is {}.".format(k, acc))

k=9
pred = find_kNN(x_train, y_train, x_test, y_test, k)
acc = find_accuracy(pred, x_test)
print("For k={} accuracy is {}.".format(k, acc))