#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np

# compare two lists by last several elements with same length
# input: valid - [l1 lists] a list consists of l1 list of labels,
#        train - [l2 lists] a list consists of l2 list of labels
# output: similarity - [float] similarity between two lists
def compareList(valid,train):
    valid_len = len(valid)
    train_len = len(train)
    if valid_len < train_len:
        similarity = getSimilarity(valid,train[(train_len-valid_len):])
    else:
        similarity = getSimilarity(valid[(valid_len-train_len):],train)
    return similarity

# calculate similarity between two lists of list of labels with same length
# similarity = sum_{i=1}^l{# of occurance elements/# of union elements}/l
# input: l1 - [l1' lists] a list consists of l1' list of labels,
#        l2 - [l2' lists] a list consists of l2' list of labels, l1' = l2' = l
# output: similarity - [float] similarity between two lists
def getSimilarity(l1,l2):
    l = len(l1)
    similarity = 0
    for d1,d2 in zip(l1,l2):
        similarity += float(len(set(d1).intersection(set(d2))))/ \
                        float(len(set(d1).union(set(d2))))
    similarity /= float(l)
    return similarity

# get votes of labels according to neighbors
# input: data - [n lists] data with n lists of labels
#        index - [i*1 array] indices of samples in data to vote for prediction
# output: v - [c*2 matrix] v[0] as the label, v[1] as the votes for the label
def getVote(data,index):
    votes = {}
    
    for i in index:
        # get the last element in the list as the prediction label list
        for code in data[i][-1]:
            # add count of each label to votes
            if code in votes:
                votes[code] += 1
            else:
                votes[code] = 1
    
    # sort labels according to [vote count descending, label number]
    votes = list(votes.items())
    v = sorted(votes,key = lambda l:(l[1],-l[0]),reverse = True)
    return v

# calculate prediction based on data and neighbors index
# input: data - [n lists] a list consists of n lists of labels,
#        index - [n lists] a list consists of n lists of index
# output: pred - [n lists] prediction from data
def getPrediction(data,index):
    pred = []
    for idx in index:
        votes = getVote(data,idx)
        pred.append(votes)
    return pred

def getNeighbors(valid_data,training_data,K=2):
    k_neighbors = [] # indices of nearest K neighbors
    n = len(training_data)
    for v in valid_data:
        # store {indices:similarity} of K nearest neighbors of sample v
        neighbors = {}
        
        # a neighbor is the sample in training_data with high similarity 
        # except for the last element (predicted labels)
        # initialization
        for k in range(K):
            neighbors[k] = compareList(v[:-1],training_data[k][:-1])
        # update neighbors
        for k in range(K,n):
            similarity = compareList(v[:-1],training_data[k][:-1])
            # replace the neighbor with minimum similarity 
            # if a new neighbor has larger similarity
            min_similarity = np.min(neighbors.values())
            if similarity > min_similarity:
                for key in neighbors.keys():
                    if neighbors[key] == min_similarity:
                        neighbors.pop(key)
                        break
                neighbors[k] = similarity
        
        neighbors = sorted(list(neighbors.items()),key=lambda x:-x[1])
        k_neighbors.append(map(lambda x:x[0],neighbors))
    return np.asarray(k_neighbors)
    
# get top K predictions from full set of label prediction
# input: pred - [m lists] a list consists of m 2d lists, the 2d list is 
#               ([label, # votes]) ordered by # votes descending
#        K - [int] select top K predicted labels with high # votes
# output: top_k - [m lists] a list consists of m lists
def getTopK(pred,K=10):
    n = len(pred)
    top_k = []
    for p in pred:
        # get top min(K,len(p)) labels as prediction
        top_k.append(map(lambda x:x[0],p[:min(K,len(p))]))
    return top_k
    
# calculate accuracy of prediction, indicating how many ground truth
#      labels are predicted
# note a ground truth is l, which is a list of labels, and prediction
#      is p, a list of labels
# the accuracy is definded as (# intersection(l,p))/(# labels in l)
# input: pred - [m lists] a list consists of m lists of labels,
#        label - [m lists] a list consists of m lists of labels
# output: accuracy - [m*1 array] a list of accuracy of prediction
def getAccuracy(pred,label):
    n = len(pred)
    accuracy = []
    for p,l in zip(pred,label):
        p,l = set(p),set(l)
        acc = float(len(l.intersection(p)))/float(len(l))
        accuracy.append(acc)
    return accuracy
    
