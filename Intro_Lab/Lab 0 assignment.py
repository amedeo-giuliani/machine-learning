#!/usr/bin/env python
# coding: utf-8

# # Hands-on exercise
# 
# In this simple example you are required to perform a simple linear regression with scipy. Find all the information on the function in the documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html

# ### Assignment
# 
# 1) Load the provided .csv file with the used car data
# 
# 2) Use a linear regression to estimate the car prices from the year, kilometers or engine power. You can make a simple 1D regression from each one of the parameters independently (as an optional task you can also try a 2D or 3D regression combining multiple cues)
# 
# 3) Firstly perform the estimation using the scipy linregress function (or alternatively you can use the sklearn.linear_model.LinearRegression class).
# NB: check the documentation of the two methods!! In particular be aware of the number of outputs (in case use "_" to avoid the return of a specific output).
# 
# 4) Have a look at the correlation coefficient to see which of the 3 features works better
# 
# 5) Then implement the least square algorithm: you should get exactly the same solution of linregress !
# 
# 6) Plot the data and the lines representing the output of the linregress and least square algorithms
# 

# In[1]:


import matplotlib.pyplot as plt
import csv
from scipy import stats
import numpy as np
import sklearn as sl
from sklearn import linear_model


# In[2]:


# Load the provided data file with the used car data (you can also have a look at it with any text editor)

filename = "data/km_year_power_price.csv"
lines = csv.reader(open(filename, newline=''), delimiter=',')

# place your loading code here
dataset = list(lines)[1:]
for i in range(len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]
dataset = np.array(dataset)
print(dataset)
print(len(dataset))
print(dataset.shape)


# Use linear regression to estimate the car prices from the year, kilometers or engine power. 
# You can make a simple 1D regression from each one of the parameters independently 
# 
# 
# 

# In[96]:


# linear regression with linregress (estimate price from year)

year = dataset[:,1]
price = dataset[:,3]
slope1,intercept1,rvalue1,_,_ = stats.linregress(year,price)


# In[ ]:


# (Optional) linear regression with linear_model.LinearRegression() (estimate price from year)
# Recall that in Python a mx1 matrix is different from a 1D array -> need to reshape

# your code.....


# In[ ]:


# (Optional) perform linear regression with a manually implemented least squares (estimate price from year)
# You should get exactly the same solution of linregress !

# your code.....


# In[91]:


# Plot the data and the lines representing the output of the linregress and least square algorithms

plt.clf()
plt.plot(year,price,'.',label='dataset')
plt.plot(year,intercept1+slope1*year,label='fit')
plt.legend()
plt.savefig('linreg_pricefromyear.jpg',dpi=480)


# In[97]:


# linear regression with linregress (estimate price from power)

power = dataset[:,2]
price = dataset[:,3]
slope2,intercept2,rvalue2,_,_ = stats.linregress(power,price)

plt.clf()
plt.plot(power,price,'.',label='dataset')
plt.plot(power,intercept2+slope2*power,label='fit')
plt.legend()
plt.savefig('linreg_pricefrompower.jpg',dpi=480)


# In[98]:


# linear regression with linregress (estimate price from km)

km = dataset[:,0]
price = dataset[:,3]
slope3,intercept3,rvalue3,_,_ = stats.linregress(km,price)

plt.clf()
plt.plot(km,price,'.',label='dataset')
plt.plot(km,intercept3+slope3*km,label='fit')
plt.legend()
plt.savefig('linreg_pricefromkm.jpg',dpi=480)


# In[59]:


# Have a look at the correlation coefficients to see which of the 3 features works better

print(rvalue1,rvalue2,rvalue3)


# In[75]:


# (Optional) 2D linear regression with linear model (estimate price from year and power)

import pandas as pd
from matplotlib.ticker import MaxNLocator

dataset = pd.read_csv(filename)
#display(dataset)
year_power = dataset[['year','powerPS']].to_numpy()
price = dataset['avgPrice'].to_numpy()
linreg = linear_model.LinearRegression()
linreg.fit(year_power,price)
slopes = linreg.coef_
X,Y = np.meshgrid(year_power[:,0],year_power[:,1])
fit = linreg.intercept_+slopes[0]*X+slopes[1]*Y

plt.rcParams["figure.figsize"]=(10,10)
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.plot(year_power[:,0],year_power[:,1],price,'.',label='dataset')
ax.plot_surface(X,Y,fit,label='fit')
plt.savefig('linreg_pricefromyearpower.jpg',dpi=480)

