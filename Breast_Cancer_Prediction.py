#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[7]:


cancer=load_breast_cancer()
cancer


# In[9]:


data=pd.DataFrame(cancer.data,columns=cancer.feature_names)


# In[10]:


data


# In[11]:


data['target']=cancer.target


# In[45]:





# In[12]:


data


# In[13]:


data.shape


# In[14]:


data.isnull().sum()


# In[15]:


data.info()


# In[16]:


data.describe()


# In[18]:


data['target'].value_counts()


# In[19]:


# 0 -> M
#1 -> B


# In[25]:


plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
sns.barplot(data)


# # Model Training

# In[46]:


x=data.drop(columns=['target'],axis=1)
inp=x.iloc[0].values
y=data['target']
y


# In[30]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
x_train


# In[33]:


linear=LogisticRegression()


# In[34]:


linear.fit(x_train,y_train)

y_pred=linear.predict(x_test)

y_pred


# In[35]:


accuracy_score(y_pred,y_test)


# In[56]:


inp=np.asarray(inp)
inp=inp.reshape(1,-1)
linear.predict(inp)


# In[ ]:




