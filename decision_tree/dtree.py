#!/usr/bin/env python

import os
import sys
import math
from collections import defaultdict
from collections import Counter
import random
import optparse
from operator import itemgetter # provide in-place sort
from mldata import parse_c45
from pprint import pprint


parser = optparse.OptionParser()
parser.add_option("-d", "--maxdepth", dest="maxdepth", type="int",
            help=("""Set the maxmium depth of the learning desicion tree. \
If this value is zero, grow the full tree. If this value is positive, grow the \
tree to the given value [default: %default]"""))
parser.add_option("-p", action="store_true",dest="printtree",
            help=("""Set the boolean type printtree, 1 for yes, 0 for no.\
[default: %default]"""))
parser.set_defaults(maxdepth=0,printtree=False)
opts, args = parser.parse_args()
assert opts.maxdepth >= 0, "Maxium depth must be at least 0."
filename = sys.argv[-1] # not safe..
data_objects = parse_c45(filename, filename) # Use relative path    
FEATURES = data_objects.schema.features # Get features
tree_size = 0 # Global 
tree_depth = 0 # Global


def main():
    
    raw_dataset = [data.features for data in data_objects]
    dataset = [data for data in raw_dataset if None not in data] # Discard data with missing attribute value
    random.shuffle(dataset)
    dataset = dataset[:10000]
    split = len(dataset)*4/5
    train_data = dataset[:split]
    test_data = dataset[split:]
    tree = dtree(train_data,opts.maxdepth)
    if opts.printtree == 1:
        pprint(tree)
    print("Accuracy: {}".format(accuracy(tree, test_data)))
    print("Size: {}".format(tree_size))
    print("Maximum Depth: {}".format(tree_depth))
    
# ===================================================
# dtree
def dtree(dataset, max_depth ,features=None, leaf=None):    
    
    global FEATURES
    global tree_size
    global tree_depth
    
    if features is None:
        features = range(1,len(FEATURES)-1) # Initialize features
        
    class_counts = Counter([data[-1] for data in dataset])

    if not dataset or not features: # If no more features
        tree_size += 1 # leaf node +1
        branch_depth = len(FEATURES)-2-len(features)
        if branch_depth > tree_depth:
            tree_depth = branch_depth
        return leaf
        
    elif len(class_counts) == 1: # All true or all false
        leaf = class_counts.most_common(1)[0][0] # most_common eg: (True:99)
        tree_size += 1 # leaf node +1
        branch_depth = len(FEATURES)-2-len(features)
        if branch_depth > tree_depth:
            tree_depth = branch_depth
        return leaf

    else: # build tree / subtree
        leaf = majority(dataset)
        if len(FEATURES)-2-len(features) == max_depth and max_depth != 0:
            tree_size += 1
            branch_depth = len(FEATURES)-2-len(features)
            if branch_depth > tree_depth:
                tree_depth = branch_depth
            return leaf
        (best, threshold) = choose_best_feature(dataset,features)
        if best == -1: # if no gain, return leaf
            tree_size += 1
            branch_depth = len(FEATURES)-2-len(features)
            if branch_depth > tree_depth:
                tree_depth = branch_depth
            return leaf
        tree = {best:{}}
        tree_size += 1 # feature node +1
        partitions = partition_data(dataset, best, threshold)
        remaining_features = [i for i in features if i != best]
        
        for feature in partitions:
            subtree = dtree(partitions[feature],max_depth,remaining_features,leaf)
            tree[best][feature] = subtree # Need more code here
            
    return tree
        
 
def entropy(dataset):
    
    class_freq = Counter([data[-1] for data in dataset])
    entropy_value = 0.0
    if len(class_freq) == 1:
        return 0
    else:
        for label in class_freq:
            freq = class_freq[label]/float(len(dataset))
            entropy_value += -freq*math.log(freq)
        return entropy_value

def information_gain(dataset, feature):
    
    f = feature
    partitions = defaultdict(list) # Store partitions
    threshold = None
    if(FEATURES[f].type == "BINARY" or 
       FEATURES[f].type == "NOMINAL"):
        entropy_after_partition = 0
        for data in dataset:
            partitions[data[f]].append(data)
        for key in partitions:
            freq = len(partitions[key])/float(len(dataset))
            entropy_after_partition += freq*entropy(partitions[key])
            
    elif FEATURES[f].type == "CONTINUOUS":
        dataset.sort(key=itemgetter(f))
        thresholds = [] # Each threshold is (threshold_value, index) 
        entropy_after_partition = 1
        
        i = 0
        while i < len(dataset)-1:
            if (dataset[i][-1] != dataset[i+1][-1]):
                thresholds.append(((dataset[i][f]+dataset[i+1][f])/2, i)) # Get the split_point
                i += int(math.ceil(len(dataset)/100.0)) # A step of 100th if class changes
            else:
                i += 1 # A step of only one if class not changes
        for each in thresholds:
            trial_entropy = 0
            partitions[0] = dataset[:each[1]+1]
            partitions[1] = dataset[each[1]+1:]
            for key in partitions:
                percentage = len(partitions[key])/float(len(dataset))
                trial_entropy += percentage * entropy(partitions[key])
            if trial_entropy < entropy_after_partition:
                entropy_after_partition = trial_entropy
                threshold = each[0]
                
    gain = entropy(dataset) - entropy_after_partition
    return (gain, threshold)
    
def choose_best_feature(dataset, features):
    
    index = -1
    best_gain = 0
    best_threshold = None
    for i in features:
        (gain, threshold) = information_gain(dataset, i)
        if best_gain < gain: # Get the attribute with best gain
            best_gain = gain
            index = i
            best_threshold = threshold
    return (index, best_threshold) # Reture best_index and threshold point if exist for continuous feature

def partition_data(dataset, feature, threshold=None):
    
    partitions = defaultdict(list)
    if(FEATURES[feature].type == "BINARY" or 
       FEATURES[feature].type == "NOMINAL"):
        for data in dataset:
            partitions[data[feature]].append(data)
    elif FEATURES[feature].type == "CONTINUOUS":
        for data in dataset:
            if data[feature] <= threshold:
                partitions[(0,threshold)].append(data) # 0 means smaller than
            else:
                partitions[(1,threshold)].append(data) # 1 means larger than
    return partitions

def majority(dataset):
    
    class_counts = Counter([data[-1] for data in dataset])
    return class_counts.most_common(1)[0][0]


def classify(tree, data, leaf=None):
    
    if not isinstance(tree, dict):
        return tree
    i = tree.keys()[0]
    v = tree.values()[0] # Is a dict
    value = data[i]
    if isinstance(v.keys()[0],tuple): # If is continuous feature    
        if v.keys()[0][0] == 0: # If key with (0, float) is ahead
            if value < v.keys()[0][1]:
                return classify(v[v.keys()[0]],data,leaf)
            else:
                return classify(v[v.keys()[0]],data,leaf)
        else:
            if value < v.keys()[0][1]:
                return classify(v[v.keys()[1]],data,leaf)
            else:
                return classify(v[v.keys()[0]],data,leaf)
    elif value not in v: # If data feature value not in tree
        return leaf
    else:
        return classify(v[value], data, leaf)

def accuracy(tree, test_data):
    
    num_correct = 0.0
    for data in test_data:
        test = classify(tree,data)
        if test == data[-1]:
            num_correct += 1
    return num_correct / len(test_data)
    
    
main()