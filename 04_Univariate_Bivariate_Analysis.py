'# In[1]  - Documenation
"""
Script: 04_Univariate_Bivariate_Analysis.py
Description: EDA Python code for data anlaysis Univariate & Bivariate
Author: Rana Pratap
Created: Jan 2021
Version: 1.0
-*- coding: utf-8 -*-
"""
print(__doc__)
#!/usr/bin/env python
# coding: utf-8

# In[2]:
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# In[3] - Import Data and anlysze
os.chdir('D:/PROJECTS/PYTHON/06_EDA_Matplotlib')
df_auto = pd.read_csv('autos.csv')

print('Summary: \n',df_auto.describe())
print('Dtype: \n',df_auto.dtypes)
print('Sample: \n',df_auto.head(10))
df_auto['price'] = df_auto['price'].astype('float')

print(df_auto['drive'].value_counts())
print(df_auto.corr())

del(os)

# In[9]: Univariate Analysis - Distplot
sns.distplot(df_auto['price'])

# In[10]: Distplot
sns.distplot(df_auto['price'],hist=False)

# In[11]: Distplot
sns.set_color_codes()
sns.distplot(df_auto['price'],color='green')

# In[12]: BoxPlot
sns.boxplot(df_auto['price'],color='red')

# In[13]: CountPlot
print(df_auto['drive'].value_counts())
sns.countplot(df_auto['drive'])

# In[21]: Bivariate Analysis - Correlation Plot
df_num = df_auto.select_dtypes(include=[np.number])
correlation = df_num.corr()
correlation['price'].sort_values(ascending=False)*100

f, ax = plt.subplots(figsize=(14,14))
plt.title('Correlation with Price',y=1,size=20)
sns.heatmap(correlation,square=True,vmin=0.2,vmax=0.8)
del(correlation,df_num)

# In[22]: RegPlot
print(df_auto[['eng_cc','price']].corr())
sns.regplot(x='eng_cc',y='price',data=df_auto)

# In[23]: RegPlot
print(df_auto[['hw_mpg','price']].corr())
sns.regplot(x='hw_mpg',y='price',data=df_auto)

# In[24]: BoxPlot
sns.boxplot(x='style',y='price',data=df_auto)

# In[25]: BoxPlot
sns.boxplot(x='eng_loc',y='price',data=df_auto)

# In[26]:
del(df_auto)
