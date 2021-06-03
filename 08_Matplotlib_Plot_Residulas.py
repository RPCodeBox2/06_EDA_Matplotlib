# In[1] - Documentation
# -*- coding: utf-8 -*-
"""
Script: 12_Scikit_Split_Model_Selection.py
Description: Matplotlib with residual lines
Author - Rana Pratap
Date Created on Thu Jun  3 16:39:30 2021
Version - 1.0
"""
print(__doc__)

# In[2] - Import libraries
import matplotlib.pyplot as plt
import numpy as np

# Import Data
x = np.linspace(-2.2,2.2,20)
y = np.sin(x)
dy = (np.random.rand(20)-0.5)*0.5

# In[3] - Plot the chart
fig, ax = plt.subplots()
ax.plot(x,y,color='red')
ax.scatter(x,y+dy,color='orange')
ax.vlines(x,y,y+dy,color='brown',linestyle='--')
plt.show()

# In[] -
del(x,y,dy)
