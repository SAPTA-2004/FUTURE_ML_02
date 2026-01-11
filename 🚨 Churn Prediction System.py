#!/usr/bin/env python
# coding: utf-8

# In[172]:


pip list


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import shapiro


# In[ ]:





# In[ ]:





# In[4]:


df=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df['PaymentMethod'].value_counts()


# In[8]:


df.dtypes


# # Handling Duplicates

# In[9]:


duplicates=df.duplicated()
df[duplicates]


# # Handling Missing Values and DataTypes
# 

# In[10]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)


# # Handling Outliers

# In[11]:


before=sns.boxplot(x='Churn', y='TotalCharges',hue='gender',data=df)


# In[12]:


from scipy.stats import boxcox
def boxcox_transformation(df,column_name):
    transformed_data,_=boxcox(df[column_name])
    df[f'{column_name}_boxcox']=transformed_data
    stat,p_value=shapiro(transformed_data)
    distribution=sns.kdeplot(transformed_data)
    print(distribution)
    print("p_value:",p_value)


# In[13]:


boxcox_transformation(df,'TotalCharges')


# In[14]:


before=sns.boxplot(x='Churn', y='TotalCharges_boxcox',hue='gender',data=df)


# In[15]:


df


# In[ ]:





# # Univariate Analysis

# In[16]:


df['Churn'].value_counts()


# In[17]:


plt.figure(figsize=(10,8))
plt.hist(df['Churn'],bins=3,color='lightgreen',rwidth=0.8)
plt.xlabel('Churn')
plt.ylabel('Count of customers')


# In[18]:


df.head()


# In[19]:


sorted_value=df.sort_values(by='TotalCharges',ascending=False)
sorted_value


# In[20]:


percentage=df['gender'].value_counts()/len(df['gender'])*100
percentage


# In[21]:


percentage.plot(kind='pie')


# In[22]:


d1=pd.crosstab(df['Churn'],df['gender'])
d1


# In[23]:


d2=pd.crosstab(df['gender'],df['Partner'])
d2


# In[24]:


d1.plot(kind='pie',subplots=True)


# # BIVARIATE ANALYSIS
# 

# In[25]:


sns.scatterplot(x=df['tenure'],y=df['TotalCharges'],hue=df['gender'],style=df['Partner'],size=df['MonthlyCharges'])


# In[26]:


pivot_table=df.pivot_table(index='Churn',values=['tenure','MonthlyCharges','TotalCharges']
                           ,aggfunc={'tenure':'mean','MonthlyCharges':'mean','TotalCharges':'mean','Churn':'count'})
pivot_table


# In[ ]:





# In[27]:


plt.figure(figsize=(10,6))
sns.countplot(x='Contract',hue='gender',data=df)
plt.show()


# In[28]:


plt.figure(figsize=(10,6))
sns.countplot(x='PaymentMethod',hue='gender',data=df)
plt.show()


# In[29]:


plt.figure(figsize=(10,6))
sns.countplot(x='InternetService',hue='gender',data=df)
plt.show()


# In[30]:


plt.figure(figsize=(10,6))
sns.barplot(x='Dependents',y='TotalCharges',hue='gender',data=df)
plt.show()


# In[31]:


plt.figure(figsize=(10,6))
sns.barplot(x='gender',y='tenure',hue='Churn',data=df)
plt.show()


# In[ ]:






# In[ ]:





# In[32]:


sns.histplot(data=df, y='MonthlyCharges', hue='Churn', kde=False)


# In[33]:


sns.histplot(data=df, y='tenure', hue='Churn', kde=False)


# In[34]:


sns.distplot(df[df['Churn']=='Yes']['MonthlyCharges'],hist=False)
sns.distplot(df[df['Churn']=='No']['MonthlyCharges'],hist=False)


# In[38]:


sns.distplot(df[df['Churn']=='Yes']['TotalCharges'],hist=False)
sns.distplot(df[df['Churn']=='No']['TotalCharges'],hist=False)


# In[ ]:





# In[97]:


df.head()


# In[ ]:






# In[168]:


df.describe()


# In[ ]:





# In[99]:


sns.boxplot(x='Churn', y='MonthlyCharges',hue='gender',data=df)


# In[135]:


sns.boxplot(x='Churn', y='TotalCharges_boxcox',hue='gender',data=df)


# In[40]:


df_new=df[['tenure','TotalCharges','MonthlyCharges','TotalCharges_boxcox']]
df_new


# In[41]:


for col in df_new.columns:
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    
    #Histogram
    axes[0].hist(df_new[col],bins=20,color='skyblue',alpha=0.7)
    axes[0].set_title(f'Histogram:{col}')
    
    ##QQ plot
    stats.probplot(df_new[col],dist='norm',plot=axes[1])
    axes[1].set_title(f'QQ plot:{col}')
    
    plt.tight_layout()
    plt.show()


# In[ ]:






# In[42]:


df.head()


# In[43]:


pd.crosstab(df['Dependents'],df['gender'])


# In[44]:


sns.pairplot(df,hue='Churn')


# In[45]:


sns.heatmap(df_new.corr(), annot=True, cmap='coolwarm')


# In[ ]:





# In[ ]:





# In[46]:


churned=df.query('Churn=="Yes"')['TotalCharges']
existing=df.query('Churn=="No"')['TotalCharges']
t_statistic,p_value=stats.ttest_ind(churned,existing)
print("p_value:",p_value)


# In[ ]:





# In[47]:


column_to_keep=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges','Churn']
df=df[column_to_keep]
df


# In[48]:


df['MultipleLines'].value_counts()


# # Predictive Analysis

# In[51]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Define your columns
categorical_columns = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

numeric_columns = ['SeniorCitizen','tenure', 'MonthlyCharges', 'TotalCharges']

# 2. Preprocess pipeline
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_columns),
    ('cat', categorical_pipeline, categorical_columns)
])

# 3. Prepare target and features
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'No': 0, 'Yes': 1})

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 5. Apply SMOTE
X_train_proc = preprocessor.fit_transform(X_train)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train_proc, y_train)

# 6. Fit Random Forest
clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X_smote, y_smote)

# 7. Evaluate
X_test_proc = preprocessor.transform(X_test)
y_pred = clf.predict(X_test_proc)

print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[52]:


def predictiveclf(model, preprocessor, input_data_dict):
    input_df = pd.DataFrame([input_data_dict])
    processed_input = preprocessor.transform(input_df)
    prediction = model.predict(processed_input)
    return "Churn" if prediction[0] == 1 else "Not Churn"


# In[53]:


input_data = {
    'gender': "Female",
    'SeniorCitizen': 0,
    'Partner': "No",
    'Dependents': "No",
    'tenure': 1,
    'PhoneService': "Yes",
    'MultipleLines': "Yes",
    'InternetService': "Fiber optic",
    'OnlineSecurity': "No",
    'OnlineBackup': "No",
    'DeviceProtection': "No",
    'TechSupport': "No",
    'StreamingTV': "Yes",
    'StreamingMovies': "Yes",
    'Contract': "Month-to-month",
    'PaperlessBilling': "Yes",
    'PaymentMethod': "Electronic check",
    'MonthlyCharges': 200,
    'TotalCharges': 9000
}


result = predictiveclf(clf, preprocessor, input_data)
print(result)


# In[ ]:





# In[54]:


import pickle
pickle.dump(clf,open('Customer_churn_analysis_clf_new.pkl','wb'))


# In[55]:


import joblib
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(clf, 'model.pkl')


# In[28]:


get_ipython().system('pip install --upgrade scikit-learn==1.7.0')


# In[30]:


import sklearn
print(sklearn.__version__)


# In[ ]:




