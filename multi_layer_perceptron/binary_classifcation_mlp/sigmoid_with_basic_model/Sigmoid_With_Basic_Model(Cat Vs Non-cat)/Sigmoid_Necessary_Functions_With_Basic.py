
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt



# ### ReLu

# In[3]:


# ReLu
def relu(x):
    
   #  Arguments:
   #x -- A scalar or numpy array of any size.
    
    r = np.maximum(0,x)
    
    return r


# ### Sigmoid

# In[4]:


# Sigmoid
def sigmoid(x):
    
    #  Arguments:
    #x -- A scalar or numpy array of any size.
    
    s = 1/(1+np.exp(-x))
    return s


# #### He initialization

# In[5]:


def He_initialize_parameters(layers_dims):
    L = len(layers_dims)
    parameters = {}
    for k in range(1,L):
        parameters['W'+str(k)] = (np.random.randn(layers_dims[k],layers_dims[k-1]))*np.sqrt(2/layers_dims[k-1])
        parameters['b' +str(k)] = np.zeros((layers_dims[k],1))
    return parameters    


# ### Compute Cost for sigmoid

# In[6]:


def compute_cost_sigmoid(AL, Y):
    m = Y.shape[1]
    J = -(1/m)*np.sum((np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL))))
    return J


# ### Forward Propagation (   linear -> relu ->linear ->relu.....->linear ->sigmoid)

# In[7]:


def forward_prop(X, parameters):   # linear -> relu ->linear ->relu.....->linear ->sigmoid
    
    L = int((1/2)*len(parameters))
    globals()['A'+ str(0)] = X
    for l in range(1, L):  # linear->relu upto L-1 layer
        globals()['Z'+ str(l)] = np.dot(parameters['W'+str(l)], (globals()['A'+ str(l-1)])) + parameters['b' +str(l)]
        globals()['A'+ str(l)] = relu((globals()['Z'+ str(l)]))
    
    globals()['Z'+ str(L)] = np.dot(parameters['W'+str(L)], (globals()['A'+ str(L-1)])) + parameters['b' +str(L)] 
    globals()['A'+ str(L)] = sigmoid((globals()['Z'+ str(L)]))                                
    # storing values in cache
    lst = []   # creating list and adding Z1, A1, W1, b1... then converting it into a tuple
    
    for l in range(1,L+1):                                  
        lst.append(globals()['Z'+ str(l)])
        lst.append(globals()['A'+ str(l)])                              
        lst.append(parameters['W' +str(l)])
        lst.append(parameters['b' +str(l)]) 
    cache = tuple(lst)
    
    return (globals()['A'+ str(L)]), cache


# ### Backward Propagation (linear -> relu ->linear ->relu.....->linear ->sigmoid)

# In[8]:


# relu ->relu ->......->relu ->sigmoid
# cache has order ( Z1, A1, W1, b1, Z2,, A2, W2, b2...)
def back_prop_without_regularization(X, Y, cache):  
    L = int((len(cache))/4)
    m = X.shape[1]
    globals()['A'+ str(0)] = X
    for i in range(1,L+1):   # retrieving the cache elements
        globals()['Z'+ str(i)]  =  cache[4*i - 4]
        globals()['A' + str(i)] =  cache[4*i - 3]
        globals()['W' +str(i)]  =   cache[4*i - 2]
        globals()['b' +str(i)]  =   cache[4*i - 1]
    # back prop equations for last element , last layer uses sigmoid
    globals()['dZ' + str(L)] = (1/m)*((globals()['A' + str(L)]) - Y) # I have proved that, for softmax and sigmoid, this equation remains same
    globals()['dW' + str(L)] = np.dot((globals()['dZ' + str(L)]), (globals()['A' + str(L -1)]).T)
    globals()['db' + str(L)] = np.sum((globals()['dZ' + str(L)]), axis = 1, keepdims = True)
    globals()['dA' + str(L-1)]  = np.dot((globals()['W' +str(i)]).T , (globals()['dZ' + str(L)]))
    
    # back prop equations for layers L-1, L-2, ....1
    for l in range(L-1, 0, -1):
        globals()['dZ' + str(l)] = np.multiply((globals()['dA' + str(l)]), np.int64((globals()['A' + str(l)])>0))
        globals()['dW' + str(l)] = np.dot((globals()['dZ' + str(l)]), (globals()['A' + str(l -1)]).T)
        globals()['db' + str(l)] = np.sum((globals()['dZ' + str(l)]), axis = 1, keepdims = True)
        globals()['dA' + str(l-1)] = np.dot((globals()['W' +str(l)]).T , (globals()['dZ' + str(l)]))

    # updating gradients in dictionary
    gradients = {}
    for i in range(1, L+1):
        gradients[('dW' +str(i))] = globals()['dW' +str(i)] 
        gradients[('db' +str(i))] = globals()['db' +str(i)]
    
    return gradients    
        


# ### Gradient Descent Update Rule

# In[9]:


def gradient_descent_update(parameters, gradients, learning_rate):
    L = int(len(parameters)/2)
    for l in range (1,L+1):
        parameters['W' +str(l)] = parameters['W' + str(l)] - learning_rate*gradients['dW' + str(l)]
        parameters['b' +str(l)] = parameters['b' + str(l)] - learning_rate*gradients['db' +str(l)]
    return parameters   


# ### Predict Accuracy with Sigmoid

# In[10]:


def predict_accuracy(X, Y, parameters):
    m = Y.shape[1]
    AL,cache = forward_prop(X,parameters)
    AL_predicted = (AL > 0.5)
    accuracy = np.sum(np.float32(AL_predicted == Y))/m
    
    return accuracy
    
    
        


# ### Basic Sigmoid Model (with Gradient Descent, no mini batch, no regularization)

# In[12]:


def basic_model_sigmoid(X, Y, learning_rate, num_iteration, layers_dims):
    
    parameters = He_initialize_parameters(layers_dims)
    costs = []
    
    for i in range(0, num_iteration):
        AL, cache  = forward_prop(X, parameters)
        cost = compute_cost_sigmoid(AL,Y)
        if i%100==0:
             print('cost after ' +str(i) +' epochs is ' + str(cost))
        if i%10 ==0:
            costs.append(cost)
        grads  = back_prop_without_regularization(X, Y, cache)
        parameters = gradient_descent_update(parameters, grads, learning_rate)
     
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters



