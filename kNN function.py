from scipy.io import loadmat
import numpy as np
from sklearn.metrics import pairwise_distances

data = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')
camID = data['camId'].flatten()
filelist = data['filelist'].flatten()
gallery_idx = data['gallery_idx'].flatten()
labels = data['labels'].flatten()
query_idx = data['query_idx'].flatten()
train_idx = data['train_idx'].flatten()

print('camID:',camID.shape)
print('filelist:',filelist.shape)
print('gallery_idx:',gallery_idx.shape)
print('labels:',labels.shape)
print('query_idx:',query_idx.shape)
print('train_idx:',train_idx.shape)

import json
with open('PR_data/feature_data.json','r')as f:
    features = json.load(f)
print(len(features), len(features[0]))

features = np.asarray(features) # each row is a feature (data instance)
print(features.shape)

train = features[train_idx-1]
query = features[query_idx-1]
qLabels = labels[query_idx-1]
gallery = features[gallery_idx-1]
gLabels = labels[gallery_idx-1]
qID = camID[query_idx-1]
gID = camID[gallery_idx-1]

#dist = []
def kNN(k,query,qLabels,qID,gallery,gLabels,gID,metric):
    G = [] #list of lists, #list = #query imgs, length of each list = #gallery imgs used
    for i in range(len(query)):
        #indices for gallery features to use
        G.append([x for x in range(len(gallery)) if not(gLabels[x]==qLabels[i] and gID[x]==qID[i])]) 
    print('G')
    
    sorted_idx=[] 
    for i in range(len(query)):
        Dist = pairwise_distances(np.reshape(query[i],(1,-1)),gallery[G[i]],metric = metric)
        sorted_idx.append(np.argsort(Dist))
    print('sorted_idx')

    NN = [arr[0,:k] for arr in sorted_idx]
    print('NN')
    
    sum = 0
    for i in range(len(query)):
        usedLabels = gLabels[G[i]] # labels of gallery images used for each query image
        if(qLabels[i] in usedLabels[NN[i]]): 
            sum += 1
    acc = sum/len(query)
    print('acc')
    
    return acc

print('working')

acc = kNN(k = 5,query=query,qLabels=qLabels,qID=qID,gallery=gallery,gLabels=gLabels,gID=gID,metric='euclidean')
print(acc)
