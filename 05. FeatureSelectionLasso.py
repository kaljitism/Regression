#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


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


lasso = Lasso()


# In[13]:


lasso.fit(x_train, y_train)


# In[14]:


coef = lasso.coef_


# In[15]:


names = list(housing.feature_names)


# In[16]:


names


# In[17]:


plt.bar(range(len(names)), coef)
plt.xticks(range(len(names)), names, rotation=60)
plt.grid(True)
plt.ylabel('Coefficients')
plt.xlabel('Features')
plt.show()

