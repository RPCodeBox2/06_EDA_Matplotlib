# In[1] - Documentation
# -*- coding: utf-8 -*-
"""
Script: 03_Matplotlib_Multi_Data.py
Description: Matplotlib code to plot Multi value data
Author: Rana Pratap
Version: 1.0
Date: Jan 2021
"""
print(__doc__)

# In[2]:
import matplotlib.pyplot as plt
import numpy as np

# In[3]: Load Data
Years = list(range(2009,2017))
Suzuki = [6.8,68.25,222.65,401.64,661.30,1110.70,1200.21,1217.02]
Renault = [24.89,46.6,89.27,130.13,150.79,191.43,222.85,216.07]
Fiat = [15.03,12.38,8.76,16.94,30.71,35.13,26.74,6.94]
print('Years: ', Years)
print('Suzuki: ', Suzuki)
print('Renault: ', Renault)
print('Fiat: ', Fiat)

# In[4]: Simple multi data graph
plt.plot(Years,Suzuki,label='Suzuki',c='r')
plt.plot(Years,Renault,label='Renault',c='b')
plt.plot(Years,Fiat,label='Fiat',c='g')
plt.legend()
plt.show()

# In[5]: Line design multi data graph
plt.plot(Years,Suzuki,label='Suzuki',c='r',ls='--')
plt.plot(Years,Renault,label='Renault',c='b',ls='-.')
plt.plot(Years,Fiat,label='Fiat',c='g',ls=':')
plt.legend()
plt.show()

# In[6]: Font Styling
plt.rcParams['font.size']=12
plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.style']='italic'
plt.rcParams['font.weight']='light'

#plt.figure(figsize=(15,10))
plt.plot(Years,Suzuki,label='Suzuki',c='r',ls='--')
plt.plot(Years,Renault,label='Renault',c='b',ls='-.')
plt.plot(Years,Fiat,label='Fiat',c='g',ls=':')
plt.grid(color='black',linestyle='-.',linewidth=0.4)
plt.legend()
plt.show()

# In[7]: Grid Pattern data and marker styles
x = np.linspace(0,20,10)
#print(x)
plt.figure(figsize=(15,10))
L1 = x
L2 = x**2
L3 = x**2.1
L4 = x**2.2
L5 = x**2.3
plt.plot(x,L1,label='L1',lw='2',marker='s',ms=10,c='red')
plt.plot(x,L2,label='L2',lw='2',marker='^',ms=10,c='blue')
plt.plot(x,L3,label='L3',lw='2',marker='o',ms=10,c='orange')
plt.plot(x,L4,label='L4',lw='2',marker='D',ms=10,c='green')
plt.plot(x,L5,label='L5',lw='2',marker='P',ms=10,c='violet')
plt.title('Nonlinear Lines',style='normal',size=24)
plt.grid(color='black',linestyle='-.',linewidth=0.4)
plt.legend(loc='upper left')
plt.show()

# In[8]: Multi graph plot - 1
y = np.arange(200)
fig, plotnumbers = plt.subplots(2,2,sharex=False,sharey=False)
plotnumbers[0,0].plot(-y)
plotnumbers[0,1].plot(y)
plotnumbers[1,0].plot(y*y)
plotnumbers[1,1].plot(-y*y)
plt.subplots_adjust(left=0,right=2.5,top=2.5,bottom=0.1)
plt.show()

# In[9]: Multi graph plot - 2 x 2 - Rows, Columns, Index
y=np.arange(100)
ax1 = plt.subplot(221) #Specify 1 - Row 2, Column 2, Index 1
plt.plot(-y*y)
ax2 = plt.subplot(222) #Specify 1 - Row 2, Column 2, Index 2
plt.plot(y*y)
ax3 = plt.subplot(223) #Specify 1 - Row 2, Column 2, Index 2
plt.plot(y*-y)
ax4 = plt.subplot(224) #Specify 1 - Row 2, Column 2, Index 2
plt.plot(-y*-y)
plt.subplots_adjust(left=0.5,right=2,top=5,bottom=3)
plt.show()

# In[10]: Multi graph plot - 1 x 2 - Rows, Columns, Index
y=np.arange(100)
ax1 = plt.subplot(121) #Specify 1 - Row 1, Column 2, Index 1
plt.plot(-y*y)
ax2 = plt.subplot(122) #Specify 1 - Row 1, Column 2, Index 2
plt.plot(y*y)
plt.subplots_adjust(left=0.5,right=2,top=5,bottom=3)
plt.show()

# In[11]: Multi graph plot - 2 x 1 - Rows, Columns, Index

plt.figure(figsize=(12,10))
plt.subplot(211)
plt.plot(Years, Suzuki, color='blue')
plt.grid()

plt.subplot(212)
plt.scatter(Years, Fiat, color='purple')
plt.grid()

plt.show()

# In[12] - Multi graph plot - 2 x 1 - Rows, Columns, Index
plt.subplot(2, 1, 1)
plt.plot(Years, Suzuki, 'o-')
plt.subplot(2, 1, 2)
plt.plot(Years, Fiat, '.-')
plt.show()

# In[13] - Multi graph plot - 2 x 1 - Rows, Columns, Index
plt.figure(figsize=(15,10))
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Car Sales')
ax1.plot(Years, Suzuki, 'o-')
ax1.set_ylabel('Suzuki Sales')
ax1.grid()
ax2.plot(Years, Fiat, '.-')
ax2.set_xlabel('Years')
ax2.set_ylabel('Fiat Sales')
ax2.grid()
fig.show()

# In[14]: 


# In[15]:

#https://matplotlib.org/tutorials/introductory/customizing.html
