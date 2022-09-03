#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd


# In[18]:


df = pd.read_csv(r'https://raw.githubusercontent.com/codebasics/py/master/ML/4_save_model/homeprices.csv')


# In[19]:


df.head()


# In[20]:


df.shape


# In[21]:


from sklearn import linear_model


# In[24]:


model = linear_model.LinearRegression()
model.fit(df[['area']],df.price)


# In[28]:


model.predict([[2300]])


# In[32]:


import joblib


# In[33]:


joblib.dump(model , 'model_joblib')


# In[35]:


model =joblib.load('model_joblib')


# In[36]:


model.coef_


# In[ ]:




