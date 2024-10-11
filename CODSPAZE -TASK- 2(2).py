#!/usr/bin/env python
# coding: utf-8

# # Sales Prediction Model:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib .pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\mural\Downloads\wps_download\car_purchasing.csv",encoding='latin1')
df.head()


# In[3]:


df.shape


# In[4]:


df.size


# In[5]:


df.memory_usage()


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


df.head()


# In[11]:


df=df.drop(['customer name', 'customer e-mail'], axis=1)


# In[12]:


df=pd.get_dummies(df, columns=['country', 'gender'], drop_first=True)


# In[13]:


df.head()


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[15]:


x=df.drop(['car purchase amount'],axis=1)
y=df['car purchase amount']


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[18]:


x_train.head()


# In[19]:


x_test.head()


# In[20]:


y_train


# In[21]:


y_test


# In[22]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[37]:


model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)


# In[38]:


y_pred = model.predict(x_test)


# In[39]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[40]:


from sklearn.ensemble import  GradientBoostingRegressor


# In[42]:


model =GradientBoostingRegressor(random_state=42)
model.fit(x_train, y_train)


# In[43]:


y_pred = model.predict(x_test)


# In[44]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[45]:


from sklearn.tree import DecisionTreeRegressor


# In[46]:


model=DecisionTreeRegressor()
model.fit(x_train,y_train)


# In[47]:


y_pred=model.predict(x_test)


# In[48]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[49]:


import xgboost as xgb
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(x_train, y_train)


# In[50]:


y_pred_xgb = xgb_model.predict(x_test)


# In[51]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[ ]:





# In[ ]:




