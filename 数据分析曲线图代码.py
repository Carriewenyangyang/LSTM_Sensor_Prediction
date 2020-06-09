#!/usr/bin/env python
# coding: utf-8

# In[74]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

epoch_x = [1,2,3,4,5,6,7,8,9,10]
loss_laptop = [5,2,2,2,2,2,2,2,2,2]
loss_pi =  [30,27,27,27,27,26,26,25,26,26]
M_loss_laptop=np.mean(loss_laptop)
V_loss_laptop=np.var(loss_laptop)
M_loss_PI=np.mean(loss_pi)
V_loss_PI=np.var(loss_pi)
print("laptop mean",M_loss_laptop)
print("laptop var",V_loss_laptop)
print("Pi mean",M_loss_PI)
print("Pi var",V_loss_PI)

a1 = [1,2,3]
a2 = [7,8,9]
a3 = [a2[i]-a1[i] for i in range(0,len(a1))] 
s=np.std(a3)
M=np.mean(a3)


plt.figure(figsize=(8,6))
plt.plot(epoch_x,loss_laptop,'',label="laptop")
plt.plot(epoch_x,loss_pi,'',label="raspberry pi")

#设置作汴州刻度
my_x_ticks = np.arange(0, 11, 1)
my_y_ticks = np.arange(0, 0.1, 0.01)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)

plt.title('loss for training')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('error_value')
plt.grid(epoch_x)
plt.show()


# In[ ]:





# In[ ]:




