#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


# In[2]:


housing = datasets.load_boston()


# In[3]:


housing.keys()


# In[4]:


housing.data[0]


# In[5]:


housing.data.shape


# In[6]:


housing.target[0]


# In[7]:


housing.target.shape


# In[8]:


housing.feature_names


# In[9]:


print(housing.DESCR)


# In[10]:


x, y = housing.data, housing.target


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)


# In[12]:


GBRegressor = GradientBoostingRegressor(learning_rate=0.01)


# In[13]:


GBRegressor.fit(x_train, y_train)


# In[14]:


predictions = GBRegressor.predict(x_test)


# In[15]:


predictions[0]


# In[16]:


y_test[0]


# In[17]:


GBRegressor.score(x_test, y_test)

