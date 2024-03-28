#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import VotingRegressor
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


est_1 = LinearRegression(normalize=True)
est_2 = KNeighborsRegressor(n_neighbors=7)
est_3 = DecisionTreeRegressor()
est_4 = Lasso(alpha=0.5)


# In[13]:


voting_reg = VotingRegressor(estimators=[
    ('Linear Regression', est_1),
    ('KNN Regression', est_2),
    ('DT Regression', est_3),
    ('Lasso Regression', est_4)
])


# In[14]:


voting_reg.fit(x_train, y_train)


# In[15]:


voting_reg.score(x_test, y_test)


# In[16]:


predictions = voting_reg.predict(x_test)


# In[17]:


r2_score(y_test, predictions)

