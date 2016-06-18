
# coding: utf-8

# ## Sequence Prediction

# ## Cleaning the data

# In[1]:

import numpy as np
import pandas as pd
from keras.layers import Activation,Dense,Dropout
from keras.models import Sequential
from keras.layers import LSTM
import matplotlib.pyplot as plt

# In[2]:

df = pd.read_csv('/home/najeeb/Desktop/Dataset/sequence/train.csv',index_col = 0)
df.info()
df.describe()
df.head()


# In[4]:

eps = 5
X = []
for series in df['Sequence']:
    s = series.split(',')
    if len(s) > eps:
        X.append(s)


# In[5]:

print len(X)
y = []


# In[6]:

for i in range(len(X)):
    temp = X[i][0:5]
    y.append(X[i][5])
    X[i] = temp


# In[7]:

X_train = np.asarray(X[0:80000])
X_test = np.asarray(X[80000:])
print X_train.shape
print X_test.shape


# In[8]:

y_train = np.asarray(y[0:80000])
y_test = np.asarray(y[80000:])
print y_train.shape
print y_test.shape


# In[9]:

print X_train[0:10]
print type(X_train)


# In[10]:

print y_train[0:10]
print type(y_train)


# In[11]:

## Reshaping the X_test and X_train array
## LSTM asks for reshaping the array
X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
X_test = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
print X_train.shape,X_test.shape


# In[ ]:




# ## Recurrent Neural Network Model
# ### To predict the next integer of a sequence

# In[13]:

model = Sequential()
model.add(LSTM(32,return_sequences = True, input_shape = np.shape(X_train)[1:]))
model.add(Activation('sigmoid'))
model.add(Dropout(0.25))
model.add(LSTM(32,return_sequences = False))
model.add(Activation('sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('relu'))


# In[110]:

model.compile(loss = 'mse',optimizer = 'rmsprop',metrics = ['accuracy'])


# In[111]:

model.fit(X_train,y_train,batch_size=128,nb_epoch = 10,verbose = 1)


# In[112]:

model.save_weights('/home/najeeb/Desktop/Dataset/sequence/Weights.hdf5')


# In[114]:

score = model.evaluate(X_test,y_test,batch_size=128,verbose = 0)


# In[115]:

print score


# In[ ]:
