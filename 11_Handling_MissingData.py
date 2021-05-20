# In[1]  - Documenation
"""
Script: Git_EDA_Handling_MissingData.py
Description: EDA Python code for data anlaysis and handling missing values with 
                visualization
Author: Rana Pratap
Created: Jan 2021
Version: 1.0
-*- coding: utf-8 -*-
"""
print(__doc__)

# In[1]: Import Libraries
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# In[2]:
#os.chdir('D:/PROJECTS/PYTHON/06_EDA_Matplotlib')
df_auto = pd.read_csv('https://github.com/RPCodeBox/EDA_Python/blob/master/autos.csv')

# In[3]: View data head and tail
df_auto.head()
df_auto.head(10)
df_auto.tail(10)
df_auto.dtypes

df_auto.describe()
##Set option to show all columns
pd.options.display.max_columns=None
df_auto.describe(include='all')

#Set option to show all Rows
pd.options.display.max_rows=None
df_auto.describe(include='all')

# In[4]: Type Conversion
df_auto.dtypes
df_auto.head(10)

df_auto['bore']=df_auto['bore'].astype('float')
df_auto['stroke']=df_auto['stroke'].astype('float')
df_auto['hp']=df_auto['hp'].astype('float')
df_auto['rpm']=df_auto['rpm'].astype('float')
df_auto.dtypes

# In[5]: Handling '?' Values --- 1
df_auto.isnull().sum()
df_auto.replace('?',np.nan,inplace=True)
df_auto.notnull().sum()

#View data in Graphs 
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(20,10))
cmp=sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
sns.heatmap(df_auto.isnull())

# In[7]: Handling 'NA' values with Median--- 2
df_auto['bore'].fillna(df_auto['bore'].median(), inplace=True)
df_auto['stroke'].fillna(df_auto['stroke'].median(), inplace=True)
df_auto['hp'].fillna(df_auto['hp'].median(), inplace=True)
df_auto['rpm'].fillna(df_auto['rpm'].median(), inplace=True)

# View data in Graphs 
plt.figure(figsize=(20,10))
cmp=sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
sns.heatmap(df_auto.isnull())

# In[11]: Handling 'NA' values with Right Replacement--- 3
df_auto.head(5)
df_auto['doors'].value_counts()
#df_auto['doors'].value_counts().idmax()
df_auto['doors'].replace(np.nan,'four',inplace=True)

# View data in Graphs 
plt.figure(figsize=(20,10))
cmp=sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
sns.heatmap(df_auto.isnull())

# In[12]: Handling 'NA' values with drop ---4
df_auto.dropna(subset=['price'], axis=0, inplace=True)
df_auto.dropna(subset=['loss'], axis=0, inplace=True)
df_auto.reset_index(drop=True)

# View data in Graphs
plt.figure(figsize=(20,10))
cmp=sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
sns.heatmap(df_auto.isnull())

# In[14]:
del (df_auto)
