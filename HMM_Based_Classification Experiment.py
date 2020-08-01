#!/usr/bin/env python
# coding: utf-8

# # Import the necessary libraries 

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,accuracy_score
from sklearn.decomposition import PCA


# # Read the train and test files. Convert them to Pandas dataframes

# In[2]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# In[3]:


df_train.head(5)


# # Compute the covariance matrix of the dataset 

# In[4]:


data_arr = df_train.iloc[:,0:561].values
cov_data = np.cov(data_arr,rowvar=False)
print(np.linalg.det(cov_data))


# # Data preprocessing before training the classifier 

# In[5]:


pca=PCA(n_components=100)
cov_pca=pca.fit(df_train.iloc[:,0:561].values)


# # Data transformation

# In[6]:


data_train_pca=cov_pca.transform(df_train.iloc[:,0:561].values)
df_train_red=pd.DataFrame(data_train_pca)


# In[7]:


df_train_red['Subject']=df_train['subject']
df_train_red['Activity']=df_train['Activity']


# # View the transformed dataset

# In[8]:


df_train_red.head()


# # Count the number of datapoints for each posture

# In[9]:


df_train_red_STAND=df_train_red[df_train_red['Activity']=='STANDING']
df_train_red_SIT=df_train_red[df_train_red['Activity']=='SITTING']
df_train_red_LAY=df_train_red[df_train_red['Activity']=='LAYING']
df_train_red_WALK=df_train_red[df_train_red['Activity']=='WALKING']
df_train_red_WALK_d=df_train_red[df_train_red['Activity']=='WALKING_DOWNSTAIRS']
df_train_red_WALK_u=df_train_red[df_train_red['Activity']=='WALKING_UPSTAIRS']

print(df_train_red_STAND.shape)
print(df_train_red_SIT.shape)
print(df_train_red_LAY.shape)
print(df_train_red_WALK.shape)
print(df_train_red_WALK_d.shape)
print(df_train_red_WALK_u.shape)


# # Transform test data

# In[10]:


df_test.dropna(inplace=True)


# In[11]:


data_test_red=cov_pca.transform(df_test.iloc[:,0:561].values)
df_test_red=pd.DataFrame(data_test_red)


# In[12]:


df_test_red['Subject']=df_test['subject']
df_test_red['Activity']=df_test['Activity']
df_test_red.head()


# In[13]:


df_test_red.shape


# In[14]:


#calculating true labels
labels_true=[]
for i in range(len(df_test_red)):
    if (df_test_red['Activity'].iloc[i]=='STANDING'):
        labels_true.append(0)
    if (df_test_red['Activity'].iloc[i]=='SITTING'):
        labels_true.append(1)
    if (df_test_red['Activity'].iloc[i]=='LAYING'):
        labels_true.append(2)   
    if (df_test_red['Activity'].iloc[i]=='WALKING'):
        labels_true.append(3) 
    if (df_test_red['Activity'].iloc[i]=='WALKING_UPSTAIRS'):
        labels_true.append(4) 
    if (df_test_red['Activity'].iloc[i]=='WALKING_DOWNSTAIRS'):
        labels_true.append(5) 
labels_true=np.array(labels_true)           
labels_true.shape


# In[21]:


from hmmlearn import hmm


# In[16]:


#implementing hmm
#since there are 6 activity so fitting hmm for each activity
def HMM_F1score(N,M,labels_true):
    
    hmm_stand=hmm.GMMHMM(n_components=N,n_mix=M,covariance_type='diag')
    hmm_sit=hmm.GMMHMM(n_components=N,n_mix=M,covariance_type='diag')
    hmm_lay=hmm.GMMHMM(n_components=N,n_mix=M,covariance_type='diag')
    hmm_walk=hmm.GMMHMM(n_components=N,n_mix=M,covariance_type='diag')
    hmm_walk_d=hmm.GMMHMM(n_components=N,n_mix=M,covariance_type='diag')
    hmm_walk_u=hmm.GMMHMM(n_components=N,n_mix=M,covariance_type='diag')

    hmm_stand.fit(df_train_red_STAND.iloc[:,0:100].values)
    hmm_sit.fit(df_train_red_SIT.iloc[:,0:100].values)
    hmm_lay.fit(df_train_red_LAY.iloc[:,0:100].values)
    hmm_walk.fit(df_train_red_WALK.iloc[:,0:100].values)
    hmm_walk_d.fit(df_train_red_WALK_d.iloc[:,0:100].values)
    hmm_walk_u.fit(df_train_red_WALK_u.iloc[:,0:100].values)

   #calculating F1 score
    labels_predict = []
    for i in range(len(df_test_red)):
        log_likelihood_value = np.array([hmm_stand.score(df_test_red.iloc[i,0:100].values.reshape((1,100))),hmm_sit.score(df_test_red.iloc[i,0:100].values.reshape((1,100))),hmm_lay.score(df_test_red.iloc[i,0:100].values.reshape((1,100))),hmm_walk.score(df_test_red.iloc[i,0:100].values.reshape((1,100))),hmm_walk_d.score(df_test_red.iloc[i,0:100].values.reshape((1,100))),hmm_walk_u.score(df_test_red.iloc[i,0:100].values.reshape((1,100)))])
        labels_predict.append(np.argmax(log_likelihood_value))    
    labels_predict = np.array(labels_predict)  

    F1 = f1_score(labels_true,labels_predict,average='micro')
    acc = accuracy_score(labels_true,labels_predict)
    return F1,acc


# In[17]:


states=np.arange(1,36,1)
states


# In[18]:


F1_value_states = []
acc_value_states = []
for i in states:
    print("HMM has been trained for num_states= {}".format(i))
    f1,acc = HMM_F1score(i,1,labels_true)
    F1_value_states.append(f1)
    acc_value_states.append(acc)
fig,ax = plt.subplots(2,1)

ax[0].plot(F1_value_states)
ax[1].plot(acc_value_states)

plt.show()


# In[19]:


f_test = []
acc_test = []

for i in range(1,6):
    f1,acc1 = HMM_F1score(3,i,labels_true)
    f_test.append(f1)
    acc_test.append(acc1)
    
fig,ax = plt.subplots(2,1)

ax[0].plot(f_test)
ax[1].plot(acc_test)

plt.show()


# In[20]:


f1_val,acc_val = HMM_F1score(3,8,labels_true)
print(f1_val)
print(acc_val)


# In[1]:


## MANASH PRATIM KAKATI
## PG CERTIFICATION IN AI & ML
## E&ICT ACADAMY, IIT GUWAHATI


# In[ ]:




