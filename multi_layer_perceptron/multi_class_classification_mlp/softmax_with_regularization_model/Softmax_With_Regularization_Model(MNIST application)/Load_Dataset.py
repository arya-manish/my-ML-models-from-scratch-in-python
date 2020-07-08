
# coding: utf-8

# In[5]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[6]:


def read_input_file():
    mnist = input_data.read_data_sets("MNIST", one_hot=True)
    X_train, Y_train,X_test, Y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    return X_train, Y_train, X_test, Y_test


# In[3]:





# In[4]:




