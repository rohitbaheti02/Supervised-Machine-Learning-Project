#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv(r"C:\Users\comp\Downloads\framingham.csv")
df.head()


# In[3]:


#Dropping Education column
df_new=df.drop('education',axis=1)


# In[4]:


df_new.head()


# ## EDA(Exploratory Data Analysis)

# In[9]:


df_new.shape


# In[10]:


df_new.describe()


# In[11]:


df_new.info()


# In[12]:


df_new.isnull().sum()


# ## Handling null values

# In[18]:


# Check Skewness to handle missing values
print('The skewness of cigsPerDay: ',df_new['cigsPerDay'].skew())
print('The skewness of BPMeds: ',df_new['BPMeds'].skew())
print('The skewness of totChol: ',df_new['totChol'].skew())
print('The skewness of BMI : ',df_new['BMI'].skew())
print('The skewness of heartRate: ',df_new['heartRate'].skew())
print('The skewness of glucose: ',df_new['glucose'].skew())


# In[21]:


#If skewness  is more than 1 we will use median to handle the missing value and if it is less than 1 and greater than -1 
# we will use mean.
df_new['cigsPerDay']=df_new['cigsPerDay'].fillna(df_new['cigsPerDay'].median())
df_new['BPMeds']=df_new['BPMeds'].fillna(df_new['BPMeds'].median())
df_new['totChol']=df_new['totChol'].fillna(df_new['totChol'].mean())
df_new['BMI']=df_new['BMI'].fillna(df_new['BMI'].mean())
df_new['heartRate']=df_new['heartRate'].fillna(df_new['heartRate'].mean())
df_new['glucose']=df_new['glucose'].fillna(df_new['glucose'].median())


# In[22]:


df_new.isnull().sum()


# In[23]:


df_new.shape


# ## Splitting data into train test split

# In[28]:


#identifying target column
y=df_new[['TenYearCHD']]
x=df_new.drop(y,axis=1)


# In[34]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[36]:


y_train.value_counts()


# Our training datset have 2871 observation in the category 0(which means that the person will not get heartattack in next 10 years),where as only 521 observation in the category of 1(which means that the person will  get heartattack in next 10 years)
# 
# If we build the model on this data we will get very high accuracy to correct classify the data into category 0 but a very low accuracy to correctly classify the data into category 1. 
# 
# Thus , we will first balance the data by bringing more data points in category 1

# ## Balancing the dataset

# In[41]:


from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=42)
#To fit this object on X_train and Y_train  we can do this using fit_resample() or using fit_resample()
x_train_new,y_train_new=sm.fit_resample(x_train,y_train) 


# In[42]:


y_train_new


# In[43]:


#lets check targeted columns got balanced or not
y_train_new.value_counts()


# ## Building Model

# In[40]:


from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression()
model=lr.fit(x_train_new,y_train_new)
print("The mode is built successfully")


# ## Predict on test case

# In[44]:


y_test['Prediction']=model.predict(x_test)


# In[45]:


y_test


# ## Model Evaluation

# In[46]:


from sklearn.metrics import accuracy_score, confusion_matrix,classification_report


# In[52]:


print(confusion_matrix(y_test['TenYearCHD'],y_test['Prediction']))


# In[48]:


print('accuracy score:',accuracy_score(y_test['TenYearCHD'],y_test['Prediction']))


# In[50]:


print("Classification:",classification_report(y_test['TenYearCHD'],y_test['Prediction']))


# In[ ]:




