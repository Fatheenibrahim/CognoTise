#!/usr/bin/env python
# coding: utf-8

# # BHARAT INTERN : Machine learning Intern
# 
# 
# 

# # TASK 1: House price prediction

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import nltk


# In[2]:


data=pd.read_csv("C:\BARATHRAJ\RECOMMENTED\BHP.csv")
data


# In[3]:


data.isnull().sum()


# In[4]:


data['size'].value_counts()


# In[5]:


def convert1(x):
    lis=str(x).split(' ')
    return int(lis[0])


# In[6]:


df=data.copy()


# In[7]:


df['size'].fillna('2 BHK',inplace=True)


# In[8]:


df['bhk']=df['size'].apply(convert1)
df


# In[9]:


df.isnull().sum()


# In[10]:


df.drop(columns=['area_type','availability','society','balcony'],inplace=True)


# In[11]:


df


# In[12]:


df.drop(columns=['size'],inplace=True)


# In[13]:


df


# In[14]:


df['total_sqft'].unique()


# In[15]:


df.isnull().sum()


# In[16]:


df['bath']=data['bath'].fillna(df['bath'].median())


# In[17]:


df


# In[18]:


df.isnull().sum()


# In[19]:


df['location'].dropna(inplace=True)


# In[20]:


df.isnull().sum()


# In[21]:


def convert2(x):
    sim=str(x).split('-')
    if len(sim)==2:
        return (float(sim[0])+float(sim[1]))/2
    try:
        return float(x)
    except:
        return None


# In[22]:


df['total_sqft']=df['total_sqft'].apply(convert2)


# In[23]:


df


# In[24]:


df['total_sqft'].unique()


# In[25]:


df.isnull().sum()


# In[26]:


df.dropna(inplace=True)


# In[27]:


df.isnull().sum()


# In[28]:


df['price_per_sqft']=df['price']*100000/df['total_sqft']
df


# In[29]:


df[((df['total_sqft']/df['bhk'])>=300)].describe()


# In[30]:


df=df[((df['total_sqft']/df['bhk'])>=300)]
df


# In[31]:


df['location']=df['location'].apply(lambda x: x.strip() if isinstance(x, str) else x)
location_1=df.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_1


# In[32]:


location_less_than_10=location_1[location_1<=10]
location_less_than_10


# In[33]:


df['location']=df['location'].apply(lambda x : 'other' if x in location_less_than_10 else x)
df


# In[34]:


df


# In[35]:


def remove_outliers_sqft (df):
    df_output = pd. DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st= np.std (subdf.price_per_sqft)
        gen_df=subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output, gen_df],ignore_index =True)
    return df_output
data_clean1=remove_outliers_sqft(df)
data_clean1


# In[36]:


def bhk_outlier_remover (df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby ('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby( 'bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape [0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats =bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df [bhk_df.price_per_sqft< (stats ['mean'])].index.values)
    return df.drop (exclude_indices, axis="index")


# In[37]:


data_clean=bhk_outlier_remover(data_clean1)
data_clean.shape


# In[46]:


df.columns


# In[39]:


df.drop(columns=['price_per_sqft'],axis=1,inplace=True)


# In[47]:


data_clean


# In[42]:


x=data_clean.drop(columns=['price'],axis=1)
y=data_clean['price']
x


# In[58]:


data_clean


# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
# x_train

# In[59]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
y_train

# model training
# In[49]:


column_trans = make_column_transformer ((OneHotEncoder (sparse=False), ['location']),remainder='passthrough')


# In[50]:


column_trans


# In[60]:


scalar=StandardScaler()


# In[61]:


linear=LinearRegression()
pipe=make_pipeline(column_trans,scalar,linear)


# In[62]:


pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
y_pred


# In[63]:


metrics.r2_score(y_test,y_pred)


# In[64]:


data_clean.columns


# In[65]:


first_element = x_test.iloc[0]
first_element


# In[66]:


lis=['Marathahalli',1095.0,2.0,2]

data = [lis]


columns = ['location', 'total_sqft', 'bath', 'bhk']


arr_data = pd.DataFrame(data, columns=columns)
arr_data


# # Linear Regression

# In[67]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = ['total_sqft', 'bhk', 'bath']
categorical_features = ['location']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])


model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

model.fit(x_train, y_train)

y_pred = model.predict(arr_data)
y_pred


# In[ ]:




