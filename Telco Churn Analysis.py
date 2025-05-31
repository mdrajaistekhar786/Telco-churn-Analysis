#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the Data

# In[2]:


telco_base_data=pd.read_csv("C:/Users/istek/Downloads/archive (1)/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[3]:


telco_base_data.head()


# In[4]:


telco_base_data.columns.values


# In[5]:


telco_base_data.dtypes


# In[6]:


telco_base_data.describe()


# In[7]:


telco_base_data['Churn'].value_counts().plot(kind='barh',figsize=(8,6))
plt.xlabel("Count",labelpad=14)
plt.ylabel("Target Variable",labelpad=14)
plt.title("Count of Target Variable per category",y=1.02);


# In[8]:


100*telco_base_data['Churn'].value_counts()/len(telco_base_data['Churn'])


# In[9]:


telco_base_data['Churn'].value_counts()


# # .Data is highly imbalanced,ratio=73.02
# .So we analyse the data with other features while taking the target values separately to get  some insights.
# 

# In[10]:


## Concise summary of the dataframe , as we have  too many columns,we are using the verbose=True mode
telco_base_data.info(verbose=True)


# In[11]:


missing = pd.DataFrame((telco_base_data.isnull().sum()) * 100 / telco_base_data.shape[0]).reset_index()
missing.columns = ['column_name', 'missing_percentage']

plt.figure(figsize=(16, 5))
ax = sns.pointplot(x='column_name', y='missing_percentage', data=missing)
plt.xticks(rotation=90, fontsize=7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()


# # Missing Data-Initial Intuition
# .Here,we do not have any missing data.
# 
# General Thumb Rules:
# 
# >For features with less missing values-can use regression to predict the missing values or fill with  the mean of the values
#      present ,depending on the feature.
#      
# >For features with very high number of missing values -it is better to drop those columns as they give very less insight
#      on analysis.
#      
# >As there's no thumb rule  on what criteria do we delete  the columns  with high number of missing values ,but generally you
#     can delete the columns,if you have more thean 30-40% of missing values.But again there's a catch here,for example ,Is_car 
#     & car_type,People  having no cars,obviously have Car_Type as NaN(Null),but that does not make this column useless,so             decisions has to taken wisely. 
#     

# # Data Cleaning
# 1.Create a copy of base data for manipulating  & Processing.
# 

# In[12]:


telco_data=telco_base_data.copy()

# 2.Total charges should be numeric amount .Let's convert it to numerical data.
# In[13]:


telco_data.TotalCharges=pd.to_numeric(telco_data.TotalCharges,errors='coerce')
telco_data.isnull().sum()


# # ##3.As we can see there are 11 missing values in TotalCharges coulmn.Let's check these records

# In[14]:


telco_data.loc[telco_data['TotalCharges'].isnull()==True]


# In[15]:


#4.Missing value  since percentage of missing value is very low so it is safe to ignore them for further processing.


# In[16]:


#Removing missing values
telco_data.dropna(how='any',inplace=True)
#telco_data.fillna(0)


# #5.Divide customer into bins based on tenure e.g for tenure ,12 months:assign a tenure group  if 1-12,for tenure between 1 to 2 yrs,
# tenure group 13-24;so on

# In[17]:


#Get the max tenure 
print(telco_data['tenure'].max())


# In[18]:


#group the tenure in bins of 12 months
labels=["{0} -{1}".format(i,i +11) for i in range(1,72,12)]
telco_data['tenure_group']=pd.cut(telco_data.tenure,range(1,80,12),right=False,labels=labels)


# In[19]:


telco_data['tenure_group'].value_counts()


# In[20]:


#Drop columns  not required for data Processing
telco_data.drop(columns=['customerID','tenure'],axis=1,inplace=True)
telco_data.head()


# # Data Exploration

# # 1. Plot distribution of individual predictors by churn

# Univariate Analysis

# In[21]:


for i , predictor in enumerate(telco_data.drop(columns=['Churn','TotalCharges','MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=telco_data, x=predictor,hue='Churn')


# 2.Convert the target variable 'Churn' in a binary numeric variable i.e Yes=1; No=0

# In[22]:


telco_data['Churn']=np.where(telco_data.Churn=='Yes',1,0)


# In[23]:


telco_data.head()


# 3.Convert all the categorical variables into dummy variables

# In[24]:


telco_data_dummies=pd.get_dummies(telco_data)
telco_data_dummies.head()


# 9.Relationship between Monthly Charges  and Total Charges

# In[25]:


sns.lmplot(data=telco_data_dummies, x='MonthlyCharges', y='TotalCharges',fit_reg=False)


# Total Charges increases as monthly charges increases...

# 10 * Churn by Monthly Charges and Total Charges
# 
# 

# In[26]:


Mth = sns.kdeplot(telco_data_dummies.MonthlyCharges[(telco_data_dummies["Churn"] == 0)],
                  color="Red", fill=True)

Mth = sns.kdeplot(telco_data_dummies.MonthlyCharges[(telco_data_dummies["Churn"] == 1)],
                  ax=Mth, color="Blue", fill=True)

Mth.legend(["No churn", "Churn"], loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title("Monthly Charges by churn")


# Build  a Correlation of all predictors with 'Churn'

# In[27]:


plt.figure(figsize=(20,8))
telco_data_dummies.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')


# # *Derived insight.
# 

# High Churn seen in case of Month to month contracts,No online security.No Tech support,First year of subscription 
# and Fibre Optics internet.

# Low Churn is seen in case of Long term contracts.Subscription without Internet Service and the customers engaged for 5+ years
# factors like Gender.Availability of phone services and  # of number of multiple lines have almost No impact on churn 
# This is also evident from the Heatmap

# In[28]:


plt.figure(figsize=(12,12))
sns.heatmap(telco_data_dummies.corr() , cmap="Paired")


# In[29]:


new_df1_target=telco_data.loc[telco_data["Churn"]==0]
new_df1_target=telco_data.loc[telco_data["Churn"]==1]


# In[30]:


def uniplot(df,col,title,hue=None):
    sns.set_style("whitegrid")
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"]=20
    plt.rcParams["axes.labelsize"]=22
    plt.rcParams["axes.labelsize"]=30
    
    
    temp=pd.Series(data=hue)
    fig, ax= plt.subplots()
    width=len(df[col].unique()) + 7+4*len(temp.unique())
    fig.set_size_inches(width,8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title('title')
    ax=sns.countplot(data=df, x=col,  order=df[col].value_counts().index,hue=hue,palette='bright')
    
    plt.show()


# In[31]:


uniplot(new_df1_target,col='Partner',title='Distribution of Gender for Churned Customers',hue='gender')


# In[32]:


uniplot(new_df1_target,col='Contract',title='Distribution of Contract for Churned Customers',hue='gender')


# In[33]:


uniplot(new_df1_target,col='TechSupport',title='Distribution of Techsupport for Churned Customers',hue='gender')


# In[34]:


uniplot(new_df1_target,col='SeniorCitizen',title='Distribution of SeniorCitizen for Churned Customers',hue='gender')


# # Conclusion

# These are some of the quick insights from this exercise:
#     1.Electronic check medium are the highest churners
#     
#     2.Contract Type-Monthly customers are more likely to churn because of no contract terms,as they are free to go customers.
#     
#     3.No Online Security,No Tech Support category are high churners
#     
#     4.Non senior citizen are high churners
#     
#     Note:There could be many more such insights ,so take this as an assignment  and try to get more insights:)

# In[37]:


telco_data_dummies.to_csv('tel_churn.csv')


# In[ ]:





# In[ ]:





# In[ ]:




