#!/usr/bin/env python
# coding: utf-8

# # Fraud Detection System:

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[2]:


df=pd.read_csv(r"C:\Users\mural\Downloads\archive (8)\creditcard.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.size


# In[6]:


df.describe()


# In[7]:


df.memory_usage()


# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


df=df.drop_duplicates()


# In[11]:


df.head()


# In[12]:


fraud = df[df['Class'] == 1] 
valid = df[df['Class'] == 0]
print('Fraud Cases: {}'.format(len(df[df['Class'] == 1]))) 
print('Valid Transactions: {}'.format(len(df[df['Class'] == 0]))) 


# In[13]:


print('Amount details of the fraudulent transaction') 
fraud.Amount.describe()


# In[14]:


print('details of valid transaction') 
valid.Amount.describe()


# In[15]:


x = df.drop(['Class'], axis = 1) 
y= df["Class"] 
print(x.shape) 


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[18]:


sc=StandardScaler()
sc.fit(x_train,y_train)


# In[19]:


x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[20]:


x_train


# In[21]:


x_test


# In[22]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[23]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,matthews_corrcoef


# In[24]:


model=RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)


# In[25]:


y_pred=model.predict(x_test)


# In[26]:


accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)


# In[27]:


print(f'accuracy :{accuracy}')
print(f'classification report:{report}')


# In[28]:


MCC=matthews_corrcoef(y_test,y_pred)
print('The vale of matthews_corrcoef is {}'.format(MCC))


# In[29]:


from sklearn.metrics import confusion_matrix


# In[30]:


LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(y_test, y_pred) 
plt.figure(figsize =(6, 4)) 
sns.heatmap(conf_matrix, xticklabels = LABELS,   yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 


# In[ ]:





# In[ ]:




