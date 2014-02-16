import project1_code as p1
import numpy as np
import matplotlib.pyplot as plt
pos1 = [1, 1]
pos2 = [2, 1]
pos3 = [2, 2]
neg1 = [-0.1, -0.1]
neg2 = [0.3, 0.1]

feature_matrix = [pos1, pos2, neg1, neg2]
labels = [1, 1, -1, -1]
import random

def synthetic_data_averager(isGood):
    posPoints = []
    negPoints = []
    if isGood:
        posPoints = set([tuple([1+random.randint(0,10) , 1+ random.randint(0,10)]) for i in range(10)])
        negPoints = set([tuple([-1-random.randint(0,10), -1 - random.randint(0,10)]) for i in range(10)])
    else:
        posPoints = set([tuple([1+random.randint(0,10) , 1+ random.randint(0,10)]) for i in range(10)])
        negPoints = set([tuple([-1-random.randint(0,10), 1 + random.randint(10,20)]) for i in range(10)])
    feature_matrix = [i for i in posPoints]
    feature_matrix.extend([i for i in negPoints])
    label = [1]*len(posPoints) + [-1]*len(negPoints)
    return feature_matrix, label

def synthetic_data_perceptron(isGood):
    random.seed(42)
    posPoints = []
    negPoints = []
    if isGood:
        posPoints = set([tuple([random.randint(0,10) , random.randint(0,10)]) for i in range(10)])
        negPoints = set([tuple([-random.randint(0,10),- random.randint(0,10)]) for i in range(10)])
    else:
        posPoints = set([tuple([random.randint(-100,100) , random.randint(0,100)]) for i in range(10)])
        negPoints = set([tuple([random.randint(-10,10), -random.randint(0,100)]) for i in range(10)])
    feature_matrix = [i for i in posPoints]
    feature_matrix.extend([i for i in negPoints])
    label = [1]*len(posPoints) + [-1]*len(negPoints)
    return feature_matrix, label

    

