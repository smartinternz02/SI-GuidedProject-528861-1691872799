#!/usr/bin/env python
# coding: utf-8

# In[ ]:


ONLINE SHOPPERS BUYING INTENTION USING MACHINE LEARNING


# In[14]:


#importing the libraries
import pandas as pd
import numpy as rip
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_scores
import pickle


# In[15]:


#reading dataset
df=pd.read_csv('online_shoppers_intention.csv')     
df.head()


# In[16]:


#data preprocessing
#Statistical Analysis
df.info()
df.describe(include='all')
#finding the missing values if any
df.isnull().sum()


# In[17]:


#plotting
plt.figure(figsize=(10,10))
plt.subplot(121)
df['Revenue'].value_counts().plot(kind='pie',autopct='%.1f%%')
plt.subplot(122)
df['VisitorType'].value_counts().plot(kind='pie',autopct='%.1f%%')


# In[18]:


#subplot
plt.figure(figsize=(16,4))

plt.subplot(131)
plt.scatter(df[ 'Administrative'], df['Administrative_Duration'], color='r')

plt.subplot(132)

plt.scatter(df[ 'Informational'],df[ 'Informational_Duration'], color='g')
plt.subplot(133)

plt.scatter(df[ 'ProductRelated'], df[ 'ProductRelated_Duration'], color='y')


# In[19]:


pd.crosstab([df['Month'],df['VisitorType']],df['Revenue'])


# In[20]:


df.describe(include='all')


# In[21]:


#checking null values
df.isnull().sum()


# In[23]:


print(sorted(df["Month"].unique()))
print(sorted(df["Weekend"].unique()))
print(sorted(df["VisitorType"].unique()))


# In[22]:


df.isnull()


# In[24]:


#Label Encoding
#It is used to convert cateogerical value into numerical format
le =LabelEncoder()
df['Month'] = le.fit_transform(df['Month']) 
df['VisitorType'] = le.fit_transform(df['VisitorType'])
df['Weekend'] = le.fit_transform(df['Weekend'])
df['Revenue'] = le.fit_transform(df['Revenue'])


# In[25]:


df


# In[26]:


print(df["Month"].unique())
print(df["Weekend"].unique())
print(df["VisitorType"].unique())


# In[48]:


df.describe(include='all')


# In[ ]:





# In[49]:


#KMeans
dfKmeans=df.drop('Revenue',axis=1)


# In[51]:


scaler=MinMaxScaler()
scaled_df=scaler.fit_transform(dfKmeans)
dfKmeans=pd.DataFrame(scaled_df,columns=dfKmeans.columns)
dfKmeans.head()


# In[56]:


#clustering KMeans
n_cluster=range(1,10,1)
sse=[]
for i in n_cluster:
    k=KMeans(n_clusters=i)
    ypred=k.fit(scaled_df)
    sse.append(k.inertia_)
sse    


# In[59]:


#univariate distribution
plt.figure(figsize=(12,6))
plt.plot(n_cluster,sse,marker='*',markersize=20)


# In[60]:


km=KMeans(n_clusters=4)
ypred=km.fit_predict(dfKmeans)


# ###### pca=PCA(n_components=2)
# dfPCA=pca.fit_transform(dfKmeans)
# dfPCA

# In[64]:


#Training model
#feature selection 
x=df.drop('Revenue',axis=1)
y=df['Revenue']
#splitting the data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)


# In[65]:


#logistic regression
def logisticReg(x_train,x_test,y_train,y_test):
    lr=LogisticRegression()
    lr.fit(x_train,y_train)
    ypred=lr.predict(x_test)
    print('**LogisticRgession**')
    print('Confusion matrix')
    print(confusion_matrix(y_test ,ypred))
    print('Classification report')
    print(classification_report(y_test,ypred))


# In[66]:


#randonforest
def randonForest(x_train,x_test,y_train,y_test):
    rf=RandomForestClassifier()
    rf.fit(x_train,y_train)
    yPred=rf.predict('**Random Classifier**')
    print('Confusion matrix')
    print(confusion_matrix(y_test ,ypred))
    print('Classification report')
    print(classification_report(y_test,ypred))


# In[ ]:





# In[ ]:





# In[27]:


def compareModel(x_train,x_test,y_train,y_test):
    logisticReg(x_train,x_test,y_train,y_test)
    print(''*100)
    randomForest(x_train,x_test,y_train,y_test)
    


# In[69]:


rf=RandomForestClassifier()
rf.fit(x_train,y_train)
yPred=rf.predict(x_test)
cv=cross_val_score(rf,x,y,cv=5)
np.mean(cv)


# In[70]:


import pickle
pickle.dump(rf,open('model.pkl','wb'))


# In[ ]:




