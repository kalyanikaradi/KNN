#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor#regression checking
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from  sklearn.metrics import accuracy_score, confusion_matrix,classification_report,recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[4]:


data=pd.read_csv('Iris.csv')


# In[5]:


data.describe()


# In[6]:


data.head()


# In[7]:


data.info()


# In[8]:


data['target']=data.target
data.head()


# In[9]:


data[data.target==1].head()


# In[10]:


data[data.target==2].head()


# In[11]:


data0=data[:50]
data1=data[50:100]
data2=data[100:]


# In[20]:


plt.xlabel('Sepal_Length')
plt.ylabel('Sepal_Width')
plt.scatter(data0['sepal_length'],data0['sepal_width'],color='green',marker='+')
plt.scatter(data1['sepal_length'],data1['sepal_width'],color='blue',marker='.')


# In[24]:


X=data.drop(['target'],axis='columns')
y=data.target


# In[25]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=1)


# In[26]:


len(X_train)


# In[27]:


len(X_test)


# In[36]:


#creat KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


# In[37]:


knn.score(X_test, y_test)


# In[39]:


#confusion matrix
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[41]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:




