#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


# In[60]:


df=pd.read_csv("CAR DETAILS.csv")


# In[61]:


import warnings
warnings.simplefilter('ignore')


# In[62]:


pd.set_option("display.max_columns", None)
pd.options.display.float_format = '{:.2f}'.format


# In[63]:


df.head()


# In[64]:


df.shape


# In[65]:


df.describe()


# In[66]:


df.info()


# In[71]:


##Visual Plots


# In[68]:


cat_feature=[feature for feature in df.columns if df[feature].dtype=='O']
print(cat_feature)


# In[69]:


cat_feature1=[feature for feature in cat_feature if len(df[feature].unique())<50]
for feature in cat_feature1 :
    plt.figure(figsize = (20, 5))
    sns.countplot(df[feature],palette="pastel")
    plt.show()


# In[81]:


plt.figure(figsize = (15, 5))
sns.lineplot(x=df['name'],y=df['selling_price'],ci=None,palette='pastel',color='Teal')
plt.show()


# In[ ]:


##checking for Null Values


# In[77]:


df. isnull().sum()


# In[80]:


duplicate_rows = df[df.duplicated()]
print("No. of duplicate rows: ", duplicate_rows.shape[0])


# In[82]:


df.drop_duplicates(inplace=True)
print("No. of rows after dropping duplicates: ", df.shape[0])


# In[83]:


##Types of feature in the data set


# In[84]:


cat_feature=[feature for feature in df.columns if df[feature].dtype=='O']
print(cat_feature)


# In[85]:


num_feature=[feature for feature in df.columns if df[feature].dtype!='O']
print(num_feature)


# In[86]:


df.head()


# In[87]:


for feature in cat_feature:
    print(f'{feature} has {len(df[feature].unique())} values')


# In[90]:


df['name'].unique()


# In[93]:


from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()
for feature in ['name', 'selling_price', 'year', 'km_driven','seller_type','fuel','transmission','owner']:
    df[feature]=encode.fit_transform(df[feature])


# In[94]:


import scipy.stats as stats
def diagnostic_plot(data, col):
    fig = plt.figure(figsize=(20, 5))
    fig.subplots_adjust(right=1.5)
    
    plt.subplot(1, 3, 1)
    sns.distplot(data[col], kde=True, color='pink')
    plt.title('Histogram')
    
    plt.subplot(1, 3, 2)
    stats.probplot(data[col], dist='norm', fit=True, plot=plt)
    plt.title('Q-Q Plot')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(data[col],color='pink')
    plt.title('Box Plot')
    
    plt.show()


# In[95]:


for col in df.columns:
   diagnostic_plot(df, col)


# In[96]:


outlier=[feature for feature in df.columns if len(df[feature].unique())>100]
print(outlier)


# In[97]:


features=[feature for feature in df.columns]
Q1 = df[outlier].quantile(0.25)
Q3 = df[outlier].quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[98]:


df = df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]
print("No. of rows remaining: ", df.shape[0])


# In[99]:


##Scaling and Spiliting


# In[104]:


x=df.iloc[:,1:]


# In[105]:


x


# In[106]:


y=df.iloc[:,[0]]


# In[107]:


y


# In[109]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[110]:


print("X_train:",x_train.shape)
print("X_test:",x_test.shape)
print("Y_train:",y_train.shape)
print("Y_test:",y_test.shape)


# In[111]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[112]:


##Creating Model with Ridge regression


# In[124]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()


# In[125]:


parameter={'alpha':[1,0.5,0.1,0.01,5,10,50,100]}
ridgecv=GridSearchCV(ridge,parameter,scoring='neg_mean_squared_error',cv=5)
ridgecv.fit(x_train,y_train)


# In[127]:


print(ridgecv.best_params_)


# In[128]:


ridge_pred=ridgecv.predict(x_test)


# In[129]:


sns.displot(ridge_pred-y_test,kind='kde')


# In[133]:


from sklearn.metrics import r2_score
score=r2_score(ridge_pred,y_test)


# In[134]:


score


# In[135]:


plt.figure(figsize = (10, 8))
corr = df.corr(method='spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
cormat = sns.heatmap(corr, mask=mask, annot=True, cmap='YlGnBu', linewidths=1, fmt=".2f")
cormat.set_title('Correlation Matrix')
plt.show()


# In[136]:


## CREATING MODEL WITH RIDGE RIGRESSION


# In[137]:


from sklearn.linear_model import Lasso


# In[138]:


lasso=Lasso()


# In[139]:


parameter={'alpha':[0.01,0.001,0.0001,0.5,1]}
lassocv=GridSearchCV(lasso,parameter,scoring='neg_mean_squared_error',cv=5)
lassocv.fit(x_train,y_train)


# In[140]:


print(lassocv.best_params_)


# In[141]:


lasso_pred=lassocv.predict(x_test)


# In[142]:


score=r2_score(lasso_pred,y_test)


# In[143]:


score


# In[144]:


from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()


# In[145]:


parameter={"n_neighbors":[1,2,3,4,5,10,20,50]}


# In[146]:


from sklearn.model_selection import GridSearchCV
knncv=GridSearchCV(knn,parameter,scoring='neg_mean_squared_error',cv=5)
knncv.fit(x_train,y_train)


# In[147]:


print(knncv.best_params_)


# In[148]:


knn_pred=knncv.predict(x_test)


# In[149]:


sns.displot(knn_pred-y_test,kind='kde')


# In[150]:


score=r2_score(knn_pred,y_test)
score


# In[151]:


from sklearn.tree import DecisionTreeClassifier


# In[152]:


treemodel=DecisionTreeClassifier()


# In[153]:


## preprunning
parameter={
    'criterion':['gini','entropy','log_loss'],
    'max_depth':[100,200,300,500]
}


# In[154]:


from sklearn.model_selection import GridSearchCV


# In[155]:


cv=GridSearchCV(treemodel,parameter,scoring='accuracy',cv=5)


# In[157]:


cv.fit(x_train,y_train)


# In[158]:


cv.best_params_


# In[159]:


y_pred=cv.predict(x_test)


# In[160]:


from sklearn.metrics import accuracy_score,classification_report


# In[161]:


score=accuracy_score(y_pred,y_test)


# In[162]:


print(score)


# In[163]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
GB=GradientBoostingRegressor()
GB.fit(x_train,y_train)
print("Training score:",GB.score(x_train,y_train))
print("Testing score:",GB.score(x_test,y_test))


# In[2]:


#Building And Fitting adaboost regression ML Model


# In[3]:


from sklearn.ensemble import AdaBoostRegressor
abr = AdaBoostRegressor(n_estimators=10000, learning_rate=1.15)


# In[ ]:


Building and fitting Support Vector Regression ML model


# In[5]:


from sklearn.svm import SVR

svr = SVR(kernel='rbf',
    degree=3,
    gamma='scale',
    coef0=0.0,
    tol=0.001,
    C=1.0,
    epsilon=0.1,
    shrinking=True,
    cache_size=200,
    verbose=False,
    max_iter=-1,)


# In[10]:


#building and fitting Random Forest Regressor ML Model


# In[11]:


from sklearn.ensemble import RandomForestRegressor
rfc= RandomForestRegressor(n_estimators=300,criterion='poisson', max_depth= 10, max_samples=0.8)


# In[ ]:


#Saving the best suitable model
We will continue with Random Forest regression ML Model. And saving that model


# In[12]:


import pickle

model = rfc
with open('final_model.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[14]:


#Loading that Ml Model


# In[15]:


# Assuming your pickle file is named 'model.pkl'
with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)


# In[28]:


#Saving the random point in pkl format


# In[36]:


model1 = rfc
with open('data.pkl', 'wb') as f:
    pickle.dump(model1, f)


# In[ ]:




