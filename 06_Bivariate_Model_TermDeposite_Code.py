# In[1]: Documentation
"""
Scirpt: 06_Bivariate_Model_TermDeposite_Code.py
Description: Code for Bivariate analysis of live data, transformation and model preperation
Author: Rana Pratap
Version: 1.0
Date: 2021
"""
print(__doc__)
# In[2]: Import Libraries and Data
import pandas as pd
#load the CSV file
df = pd.read_csv('bank-additional-full.csv')
print ('Number of samples: ',len(df))
print (df.info())

#count the number of rows for each type
df.groupby('y').size()

# In[3]: Calculate Prevalence
df['OUTPUT_LABEL'] = (df.y == 'yes').astype('int')
def calc_prevalence(y_actual):
    # this function calculates the prevalence of the positive class (label = 1)
    return (sum(y_actual)/len(y_actual))
print('prevalence of the positive class: %.3f'%calc_prevalence(df['OUTPUT_LABEL'].values))

# In[4]: Exploring the data set and unique values
print (df.head())
df[list(df.columns)[:10]].head()
df[list(df.columns)[10:]].head()
df.info()
print('Number of columns:',len(df.columns))

def printUniqueValues(data):
    # for each column
    for a in list(data.columns):
        # get a list of unique values
        n = df[a].unique()
        if len(n)<20:
            print(a + ': ' +str(len(n)) + ' unique values -')
            print(n)
        else:
            print(a + ': ' +str(len(n)) + ' unique values')

printUniqueValues(df)


# In[5] - Numerical Features
cols_num = ['campaign', 'pdays','previous', 'emp.var.rate', 'cons.price.idx','cons.conf.idx', 'nr.employed','age','euribor3m']
df[cols_num].head()       

# In[6] - Graphical Representation of Numerical Features
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import warnings

# In[7] - Age Plot
warnings.filterwarnings('ignore')
fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'age', data =  df[cols_num])
ax.set_xlabel('Age', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Age Count Distribution', fontsize=15)
sns.despine()

# In[8] - Age Distribution
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.boxplot(x = 'age', data = df[cols_num], orient = 'v', ax = ax1)
ax1.set_xlabel('People Age', fontsize=15)
ax1.set_ylabel('Age', fontsize=15)
ax1.set_title('Age Distribution', fontsize=15)
ax1.tick_params(labelsize=15)

sns.distplot(df[cols_num]['age'], ax = ax2)
sns.despine(ax = ax2)
ax2.set_xlabel('Age', fontsize=15)
ax2.set_ylabel('Occurence', fontsize=15)
ax2.set_title('Age x Ocucurence', fontsize=15)
ax2.tick_params(labelsize=15)

plt.subplots_adjust(wspace=0.5)
plt.tight_layout()

# In[9] - Previous
fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'previous', data = df[cols_num])
ax.set_xlabel('Previous', fontsize=16)
ax.set_ylabel('Number', fontsize=16)
ax.set_title('Previous', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()

# In[10] - Emp Var Rate
fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'emp.var.rate', data = df[cols_num])
ax.set_xlabel('Emp.var.rate', fontsize=16)
ax.set_ylabel('Number', fontsize=16)
ax.set_title('Emp.var.rate', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()

# In[11] - Cons.conf.idx 
fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'cons.conf.idx', data = df[cols_num])
ax.set_xlabel('Cons.conf.idx', fontsize=16)
ax.set_ylabel('Number', fontsize=16)
ax.set_title('Cons.conf.idx', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()

#df[cols_num].isnull().sum()

# In[12] - Categorical Features and Encoding
cols_cat = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
df[cols_cat].isnull().sum()

cols_cat = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
df[cols_cat]
cols_new_cat=pd.get_dummies(df[cols_cat],drop_first = False)
cols_new_cat.head()

# In[13] - Graphical Representation of Categorical Features
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# In[14] - Education Plot
fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'education', data = df[cols_cat])
ax.set_xlabel('Education Receieved', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
ax.set_title('Education', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()

# In[15] - Martial Status 
fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'marital', data = df[cols_cat])
ax.set_xlabel('Marital Status', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
ax.set_title('Marital', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()

# In[16] - No of Jobs
fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'job', data = df[cols_cat])
ax.set_xlabel('Types of Jobs', fontsize=16)
ax.set_ylabel('Number', fontsize=16)
ax.set_title('Job', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()

# In[17] - Previous Details
fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
sns.countplot(x = 'poutcome', data = df[cols_cat])
ax.set_xlabel('Previous Marketing Campaign Outcome', fontsize=16)
ax.set_ylabel('Number of Previous Outcomes', fontsize=16)
ax.set_title('poutcome (Previous Marketing Campaign Outcome)', fontsize=16)
ax.tick_params(labelsize=16)
sns.despine()

# In[18] - Default, Housing, Loan - Set
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (20,8))
order = ['no', 'unknown', 'yes']
sns.countplot(x = 'default', data = df[cols_cat], ax = ax1, order = order)
ax1.set_title('Default', fontsize=15)
ax1.set_xlabel('')
ax1.set_ylabel('Count', fontsize=15)
ax1.tick_params(labelsize=15)

sns.countplot(x = 'housing', data = df[cols_cat], ax = ax2, order = order)
ax2.set_title('Housing', fontsize=15)
ax2.set_xlabel('')
ax2.set_ylabel('Count', fontsize=15)
ax2.tick_params(labelsize=15)

sns.countplot(x = 'loan', data = df[cols_cat], ax = ax3, order = order)
ax3.set_title('Loan', fontsize=15)
ax3.set_xlabel('')
ax3.set_ylabel('Count', fontsize=15)
ax3.tick_params(labelsize=15)

plt.subplots_adjust(wspace=0.25)

# In[19] - Contact, Month, Week - Set
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (15,6))
sns.countplot(df[cols_cat]['contact'], ax = ax1)
ax1.set_xlabel('Contact', fontsize = 10)
ax1.set_ylabel('Count', fontsize = 10)
ax1.set_title('Contact Counts')
ax1.tick_params(labelsize=10)

sns.countplot(df[cols_cat]['month'], ax = ax2, order = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
ax2.set_xlabel('Months', fontsize = 10)
ax2.set_ylabel('')
ax2.set_title('Months Counts')
ax2.tick_params(labelsize=10)

sns.countplot(df[cols_cat]['day_of_week'], ax = ax3)
ax3.set_xlabel('Day of Week', fontsize = 10)
ax3.set_ylabel('')
ax3.set_title('Day of Week Counts')
ax3.tick_params(labelsize=10)

plt.subplots_adjust(wspace=0.25)

# In[20] - In order to add the one-hot encoding columns to the dataframe, we use the concat function. axis = 1 is used to add the columns. 
df = pd.concat([df,cols_new_cat], axis = 1)
cols_all_cat=list(cols_new_cat.columns)
df[cols_all_cat].head()

# In[21] - Summary of Features Engineering
print('Total number of features:', len(cols_all_cat + cols_num))
print('Numerical Features:',len(cols_num))
print('Categorical Features:',len(cols_all_cat))

df[cols_num+cols_all_cat].isnull().sum().sort_values(ascending = False)
cols_input = cols_num + cols_all_cat
df_data = df[cols_input + ['OUTPUT_LABEL']]
cols_input
print(len(cols_input))
df_data.head(6)

# In[22] - Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x = df.drop('y',axis=1)
y = df['y']
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
del(x,y)

# In[23] - Feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)

# In[24] - Fitting Decision Tree classifier to the training set 
from sklearn.tree import DecisionTreeClassifier  
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
classifier.fit(x_train, y_train)

# In[25] - Predicting the test set result
from sklearn.metrics import confusion_matrix, classification_report
y_pred= classifier.predict(x_test)
print(classification_report(y_test,y_pred))

# In[26] - Creating the Confusion matrix  
cm= confusion_matrix(y_test, y_pred) 
print(cm)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()
del(cm,i,j)

# In[27] - Fitting Random forest classifier to the training set 
from sklearn.ensemble import RandomForestClassifier 
classifier= RandomForestClassifier(criterion='gini',n_estimators=5,
                                 random_state=1, n_jobs=2)  
classifier.fit(x_train, y_train)

# In[28] - Predicting the test set result
from sklearn.metrics import confusion_matrix, classification_report
y_pred= classifier.predict(x_test)
print(classification_report(y_test,y_pred))
cm= confusion_matrix(y_test, y_pred) 
print(cm)

# In[29] - Clear
del(x_train, x_test, y_train, y_test)
