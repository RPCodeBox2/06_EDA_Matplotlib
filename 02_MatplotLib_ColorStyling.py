# In[1] - Documentation
# -*- coding: utf-8 -*-
"""
Script: 02_MatplotLib_ColorStyling.py
Description: Python code explaining customiztion or styling
Author: Rana Pratap
Version: 1.0
Date: Jan 2021
"""
print(__doc__)

# In[2] - Impport Libraries
import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np

# In[3] - Create a data frame
x = range(1,101)
y = np.random.randn(100)*15 + range(1,101)

# In[4] - Plot the graph - Line
#plt.plot('x','y',data=df, linestyle='none', marker='o')
plt.plot(x,y,marker='')
plt.show()

# In[5] - Plot the graph - Scatter
#plt.plot('x','y',data=df, linestyle='none', marker='o')
plt.plot(x,y,linestyle='none',marker='o',color='mediumvioletred')
plt.show()

# In[6] - Plot the graph - Line and Markers
plt.figure(figsize=(15,10))
plt.plot(x,y,marker='o',color='mediumvioletred')
plt.show()

# In[7] - Plot the graph
plt.figure(figsize=(15,10))
plt.plot(x,y,linestyle='none', marker='o',color='mediumvioletred')
# Annotate with text + Arrow
plt.annotate(
# Label and coordinate
'This point is interesting!', xy=(27, 54), xytext=(0, 80),
# Custom arrow
arrowprops=dict(facecolor='red', shrink=0.05))
plt.show()

# In[8] - Bar Chart
# Make a fake dataset
height = [3, 12, 5, 18, 45]
bars = ('A', 'B', 'C', 'D', 'E')
y_pos = np.arange(len(bars))
plt.bar(y_pos, height, color=['black', 'red', 'green', 'blue', 'yellow'],
        edgecolor='black')
plt.xticks(y_pos, bars)
plt.show()

# In[5] - Clear
del(x,y)
del(height,bars,y_pos)
#https://python-graph-gallery.com/

