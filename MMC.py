#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import loadmat
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
import random
from metric_learn import MMC_Supervised


# In[2]:


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


# #   UNCOMMENT!!!!!!!!!!!!1
# with open('PR_data/feature_data.json','r')as f:
#     features = json.load(f)
# print(len(features), len(features[0]))
# 
# features = np.asarray(features) # each row is a feature (data instance)
# print(features.shape)

# In[3]:


try:
    features = loadmat('PR_data/features.mat')
    features = features['features']
except FileNotFoundError:
    print('exception handling')
    with open('PR_data/feature_data.json','r')as f: 
        features = json.load(f) 
        features = np.asarray(features) # each row is a feature (data instance) print(features.shape)
print(features.shape)


# In[4]:


def kNN(k,query,qLabels,qID,gallery,gLabels,gID,metric):
    G = [] #list of lists, #list = #query imgs, length of each list = #gallery imgs used
    for i in range(len(query)):
        #indices for gallery features to use
        G.append([x for x in range(len(gallery)) if not(gLabels[x]==qLabels[i] and gID[x]==qID[i])]) 
    print('G')
    
    sorted_idx = []
    for i in range(len(query)):
        Dist = distance.cdist(np.reshape(query[i],(1,-1)),gallery[G[i]],metric = metric)
        sorted_idx.append(np.argsort(Dist))
    print('sorted_idx')

    def accuracy(k):
        NN = [arr[0,:k] for arr in sorted_idx]

        sum = 0
        for i in range(len(query)):
            usedLabels = gLabels[G[i]] # labels of gallery images used for each query image
            if(qLabels[i] in usedLabels[NN[i]]): 
                sum += 1
        acc = sum/len(query)
        return acc
    print('NN')
    
    if type(k) is list:
        acc = []
        for n in k:
            acc.append(accuracy(n))
    else:
        acc = accuracy(k)
    
    return acc

print('working')


# In[5]:


def fit_model(model, training_dataset, query_dataset, gallery_dataset):
    model = model.fit(training_dataset.features_array(), training_dataset.labels())
    omega_query = model.transform(query_dataset.features_array())
    omega_gallery = model.transform(gallery_dataset.features_array())
    
    return model, omega_query, omega_gallery


# In[6]:


train = features[train_idx-1]
scalar = StandardScaler().fit(train)
#train = StandardScaler().fit_transform(features[train_idx-1])
train = scalar.transform(train)
tLabels = labels[train_idx-1]
#query = features[query_idx-1]
query = scalar.transform(features[query_idx-1])
qLabels = labels[query_idx-1]
#gallery = features[gallery_idx-1]
gallery = scalar.transform(features[gallery_idx-1])
gLabels = labels[gallery_idx-1]
tID = camID[train_idx-1]
qID = camID[query_idx-1]
gID = camID[gallery_idx-1]

c = len(np.unique(tLabels)) #767
print(c)

print(len(train)-c) # 6601


# In[7]:


class Datapoint(object):
    def __init__(self, features, label, cam_id):
        self.features = features
        self.label = label
        self.cam_id = cam_id

class Dataset(object):
    def __init__(self, datapoint_list=None):
        if datapoint_list:
            self.datapoints = datapoint_list
        else:
            self.datapoints = []
    
    def features_array(self):
        return np.array([datapoint.features for datapoint in self.datapoints])
    def labels(self):
        return np.array([datapoint.label for datapoint in self.datapoints])
    def cam_ids(self):
        return np.array([datapoint.cam_id for datapoint in self.datapoints])

training_dataset = Dataset([Datapoint(train[i], tLabels[i], tID[i]) for i in range(len(train))])
query_dataset = Dataset([Datapoint(query[i], qLabels[i], qID[i]) for i in range(len(query))])
gallery_dataset = Dataset([Datapoint(gallery[i], gLabels[i], gID[i]) for i in range(len(gallery))])


# In[ ]:





# In[8]:


# Create validation set
np.random.seed(seed = 342)
valLabels = np.random.choice(np.unique(tLabels),115,replace=False)
validation_subset = Dataset()
training_subset = Dataset() #6251 features
for x in training_dataset.datapoints:
    if x.label in valLabels:
        validation_subset.datapoints.append(x)
    else:
        training_subset.datapoints.append(x)


# In[9]:


#split validation_subset into validation_query and validation_gallery
def gallery_query_split(dataset):
    random.seed(3)
    shuffled = dataset
    random.shuffle(shuffled.datapoints)

    query = Dataset()
    gallery = Dataset()
    query_set = set()
    for x in shuffled.datapoints:
        if (x.label, x.cam_id) in query_set:
            gallery.datapoints.append(x)
        else:
            query.datapoints.append(x)
            query_set.add((x.label, x.cam_id))

    print(len(query.datapoints),len(gallery.datapoints))
    return query, gallery

training_query, training_gallery = gallery_query_split(training_dataset)
validation_query, validation_gallery = gallery_query_split(validation_subset)


# In[10]:


M_pca = 1000

pca = PCA(n_components=M_pca)
pca, omega_q_pca, omega_g_pca = fit_model(pca, training_subset, training_query, training_gallery)
w_pca = pca.components_
print(w_pca.shape)
np.sum(pca.explained_variance_ratio_)

acc_pca = kNN(k = [1,2,3,5,10,20,50,75,100],query=omega_q_pca,qLabels=training_query.labels(),
              qID=training_query.cam_ids(),gallery=omega_g_pca,
              gLabels=training_gallery.labels(),gID=training_gallery.cam_ids(),metric='euclidean')
print('pca training :',acc_pca)


# In[11]:


M_pca = 1000

pca = PCA(n_components=M_pca)

pca, omega_q_pca, omega_g_pca = fit_model(pca, training_subset, validation_query, validation_gallery)
w_pca = pca.components_
print(w_pca.shape, omega_q_pca.shape)
np.sum(pca.explained_variance_ratio_)

acc_pca = kNN(k = [1,2,3,5,10,20,50,75,100],query=omega_q_pca,qLabels=validation_query.labels(),
              qID=validation_query.cam_ids(),gallery=omega_g_pca,
              gLabels=validation_gallery.labels(),gID=validation_gallery.cam_ids(),metric='euclidean')
print('pca validation :',acc_pca)


# In[12]:


M_lda = 300

lda = LinearDiscriminantAnalysis(n_components=M_lda)
lda, omega_q_lda, omega_g_lda = fit_model(lda, training_dataset, query_dataset, gallery_dataset) #(1400,Mlda),(5328,M_pca)
w_lda = lda.coef_[:M_lda,:]
print(w_lda.shape)
print(np.sum(lda.explained_variance_ratio_))
print(omega_q_lda.shape,omega_g_lda.shape)

acc_lda = kNN(k = [1,2,3,5,10,20,50,75,100],query=omega_q_lda,qLabels=qLabels,qID=qID,gallery=omega_g_lda,
          gLabels=gLabels,gID=gID,metric='euclidean')
print('lda:',acc_lda)


# In[ ]:


mmc = MMC_Supervised(num_constraints=200)
mmc.fit(training_dataset.features_array(), training_dataset.labels())
omega_query, omega_gallery = mmc.transform(query_dataset.features_array()), mmc.transform(gallery_dataset.features_array())

acc_MMC = kNN(k = [1,2,3,5,10,20,50,75,100],query=omega_query,qLabels=qLabels,qID=qID,gallery=omega_gallery,
          gLabels=gLabels,gID=gID,metric='euclidean')
print('MMC:',acc_lda)


# In[ ]:




