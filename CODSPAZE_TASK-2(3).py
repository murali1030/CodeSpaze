#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis of Product Reviews:

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv(r"C:\Users\mural\Downloads\Restaurant_Reviews.tsv",delimiter='\t')
df.head()


# In[4]:


df.shape


# In[5]:


df.size


# In[6]:


df.memory_usage()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


df=df.drop_duplicates()


# In[11]:


df.head()


# In[12]:


df.duplicated().sum()


# In[13]:


import  nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# In[14]:


nltk.download('stopwords')
nltk.download('punkt')


# In[15]:


stop_words=set(stopwords.words('english'))
ps=PorterStemmer()


# In[16]:


def preprocess(text):
    text=text.lower()
    tokens=word_tokenize(text)
    tokens=[word for word in tokens if word.isalnum()]
    tokens=[word for word in tokens if word not in stop_words]
    tokens=[ps.stem(word)for word in tokens]
    return ' '.join(tokens)


# In[17]:


df['processed_Review']=df['Review'].apply(preprocess)
print(df[['Review','processed_Review']])


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[19]:


tfidf_Vectorizer=TfidfVectorizer()
X=tfidf_Vectorizer.fit_transform(df['processed_Review'])
y=df['Liked']


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[22]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)


# In[23]:


from sklearn.metrics import accuracy_score,classification_report,matthews_corrcoef


# In[24]:


y_pred=model.predict(X_test)


# In[25]:


accuracy=accuracy_score(y_test,y_pred)


# In[26]:


report=classification_report(y_test,y_pred)


# In[27]:


print(f'Accuracy:{accuracy}')
print(f'Classification Report:{report}')


# In[28]:


from sklearn.naive_bayes import MultinomialNB


# In[29]:


model=MultinomialNB()
model.fit(X_train,y_train)


# In[30]:


y_pred=model.predict(X_test)


# In[31]:


accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)


# In[32]:


print(f'Accuracy:{accuracy}')
print(f'Classification Report:{report}')


# In[33]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[34]:


model=RandomForestClassifier(n_estimators=42)
model.fit(X_train,y_train)


# In[35]:


y_pred=model.predict(X_test)


# In[36]:


accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)


# In[37]:


print(f'Accuracy:{accuracy}')
print(f'Classification Report:{report}')


# In[38]:


model=GradientBoostingClassifier(n_estimators=42)
model.fit(X_train,y_train)


# In[39]:


y_pred=model.predict(X_test)


# In[40]:


accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)


# In[41]:


print(f'Accuracy:{accuracy}')
print(f'Classification Report:{report}')


# In[42]:


MCC=matthews_corrcoef(y_test,y_pred)
print('The matthews_corrcoef is {}'.format(MCC))


# In[43]:


from sklearn.metrics import confusion_matrix ,ConfusionMatrixDisplay


# In[44]:


y_pred=model.predict(X_test)


# In[45]:


cm=confusion_matrix(y_test,y_pred,labels=model.classes_)


# In[46]:


cm=confusion_matrix(y_test,y_pred)


# In[47]:


print(f'Confusion Matrix:{cm}')


# In[48]:


disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)


# In[49]:


disp.plot(cmap=plt.cm.Blues)


# In[ ]:




