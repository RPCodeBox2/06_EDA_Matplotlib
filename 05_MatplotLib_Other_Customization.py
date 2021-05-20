# In[1] - Documentation
# -*- coding: utf-8 -*-
"""
Script: 05_MatplotLib_Other_Customization.py
Description: Advanced graphs demo 
Author: Rana Pratap
Version: 1.0
Date: 2021
"""
print(__doc__)

# In[2] - Import Libraries
import matplotlib.pyplot as plt
import numpy as np

# In[3] - Initialize Variables
x = np.arange(1,12)
y = [10,20,20,40,50,55,50,60,55,60,65]

# In[4] - Plot the graph with Text and Annotation
plt.figure(figsize=(15,10))
plt.plot(x,y,'g-.',marker='o')
plt.text(4.2,53,'My Text 1',size=16,color='purple')
plt.annotate('My Text 2', size=16, color='blue', xy=(5,50), xytext=(6,40),
             arrowprops=dict(facecolor='red', shrink=0.05), )
plt.xlabel("Values",size=18)
plt.ylabel("Measure",size=18)
plt.title('Value Sequence Graph',style='normal',size=24,color='red')
plt.grid()
plt.savefig('MyGraph01.jpg')
plt.show()

del(x,y)

# In[5]: Insert graph with Arrow included
x = np.linspace(0,20,50)
xcos = np.cos(x**2)
xsquare = x**2
fig,ax1 = plt.subplots()
plt.subplots_adjust(left=0.5,right=2,top=1.6,bottom=0)

#Setting the inset plot dimensions
left, bottom, width, height = [0.65, 1, 0.45, 0.35]
inset = fig.add_axes([left,bottom,width,height])
ax1.plot(x,xcos)
inset.plot(x,xsquare,c='red',ls=":")
ax1.annotate('Quadratic Function',xy=(8.1,0.65),xytext=(15,1),fontsize=12,
             fontweight='bold',arrowprops=dict(linewidth=2,arrowstyle='->'),rotation=0)
plt.show()

del(left, bottom, width, height)
del(x,xcos,xsquare)

# In[5]
