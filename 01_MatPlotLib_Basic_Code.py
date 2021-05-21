
# In[1] - Documentation
"""
Script - 01_MatPlotLib_Basic_Code.py
Decription - Basic coding pattern of Matplotlib
Author - Rana Pratap
Date - 2021
Version - 1.0
#!/usr/bin/env python
#coding: utf-8
"""
print(__doc__)

# In[2]: Import and Jupyter Notebook Settings
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline #Schema for Matplotlob')

# In[3]: Base Syntax
plt.plot([1,2,3,4,5])

# In[4]: Base Syntax, X axis and Label
plt.plot([1,2,3,4,5])
plt.ylabel("Sequence Numbers")
plt.show()

# In[5]: Base Syntax, X and Y axis with Labels
x = [1,2,3,4,5]
y = [10,20,20,40,50]
plt.plot(x,y)
plt.xlabel("Values")
plt.ylabel("Measure")
plt.show()

# In[6]: Base plot with line color and Pattern 1
x = [1,2,3,4,5]
y = [10,20,20,40,50]
plt.plot(x,y,'g--')

# In[7]:  Base plot with line color and Pattern 2
x = [1,2,3,4,5]
y = [10,20,20,40,50]
plt.plot(x,y,'b--')

# In[8]:  Base plot with line color and Pattern 3
x = [1,2,3,4,5]
y = [10,20,20,40,50]
plt.plot(x,y,'r-.')


# In[9]: NumPy data set and MatPlot graphs
import numpy as np
data = np.arange(0,10,0.5)
data

# In[10]: Base plot with line color and Pattern 4
plt.plot(data,data*2,'y--')

# In[11]: Base plot with multi axis plot and designs
data = np.arange(0,6,0.2)
data
plt.figure(figsize=(10,10))
plt.plot(data,data,'r--',data,data**2,'b-.',data,data**3,'g^')
plt.grid(color='black',linestyle='-.',linewidth=0.4)
plt.show()

# In[12]: Plot X and Y with Grid patten
x = [1,2,3,4,5]
y = [10,20,20,40,50]

plt.figure(figsize=(10,10))
L1 = x
plt.plot(x,y,label='L1',lw='1',marker='^',ms=10,c='blue')
plt.title('Nonlinear Lines',style='normal',size=24)

plt.grid(color='black',linestyle='-.',linewidth=0.4)

plt.legend(loc='upper left')
plt.show()

# In[29]:
del(x,y,data)
del(L1)
