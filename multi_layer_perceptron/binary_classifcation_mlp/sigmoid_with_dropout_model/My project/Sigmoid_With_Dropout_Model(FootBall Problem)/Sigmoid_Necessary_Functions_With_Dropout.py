
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt


# ### ReLu

# In[19]:


# ReLu
def relu(x):
    
   #  Arguments:
   #x -- A scalar or numpy array of any size.
    
    r = np.maximum(0,x)
    
    return r


# ### Sigmoid

# In[20]:


# Sigmoid
def sigmoid(x):
    
    #  Arguments:
    #x -- A scalar or numpy array of any size.
    
    s = 1/(1+np.exp(-x))
    return s


# ### Parameters Initialization

# In[21]:


def He_initialize_parameters(layers_dims):
    np.random.seed(3)
    L = len(layers_dims)
    parameters = {}
    for k in range(1,L):
        parameters['W'+str(k)] = (np.random.randn(layers_dims[k],layers_dims[k-1]))*np.sqrt(2/layers_dims[k-1])
        parameters['b' +str(k)] = np.zeros((layers_dims[k],1))
    return parameters    


# ### Compute Cost for sigmoid (same as for basic sigmoid)

# In[22]:


def compute_cost_sigmoid(AL, Y):
    m = Y.shape[1]
    J = -(1/m)*np.nansum((np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL))))                                   
    return J


# In[23]:


#Z = np.random.randn(1,10)
#AL = sigmoid(Z)
#Y = np.array([[1,0,1,1,1,0,0,1,0,1]])
#print(AL.shape)
#print(Y.shape)
#lambd = 0.6
#layers_dims = [100,20,15,1]
#parameters=  He_initialize_parameters(layers_dims)
#cost = regularized_compute_cost_sigmoid(AL, Y, parameters, lambd)
#print(cost)


# ### Forward Propagation (   linear -> relu ->linear ->relu.....->linear ->sigmoid) with Dropout

# In[24]:


def forward_prop_with_dropout(X, parameters, keep_prob):   # linear -> relu ->linear ->relu.....->linear ->sigmoid
    
    np.random.seed(1)
    
    L = int((1/2)*len(parameters))
    globals()['A'+ str(0)] = X
    for l in range(1, L):  # linear->relu upto L-1 layer
        globals()['Z'+ str(l)] = np.dot(parameters['W'+str(l)], (globals()['A'+ str(l-1)])) + parameters['b' +str(l)]
        globals()['A'+ str(l)] = relu((globals()['Z'+ str(l)]))
        # dropout equations on the hidden layers and not on the input and output layers
        globals()['DA' + str(l)] = np.random.rand((globals()['A'+str(l)]).shape[0],(globals()['A'+str(l)]).shape[1]) # DA1, DA2 etc. stands for dropout A1, A2 etc   
        globals()['DA' + str(l)] = globals()['DA' + str(l)] < keep_prob
        globals()['A'+ str(l)] = np.multiply(globals()['A'+ str(l)], globals()['DA' + str(l)])
        globals()['A'+ str(l)] =  (globals()['A'+ str(l)]) /keep_prob
        
    globals()['Z'+ str(L)] = np.dot(parameters['W'+str(L)], (globals()['A'+ str(L-1)])) + parameters['b' +str(L)] 
    globals()['A'+ str(L)] = sigmoid((globals()['Z'+ str(L)]))                                
    # storing values in cache
    lst = []   # creating list and adding Z1, A1, W1, b1... then converting it into a tuple
    
    for l in range(1,L):                                  
        lst.append(globals()['Z'+ str(l)])
        lst.append(globals()['DA'+ str(l)])
        lst.append(globals()['A'+ str(l)])
        lst.append(parameters['W' +str(l)])
        lst.append(parameters['b' +str(l)])
    lst.append(globals()['Z'+ str(L)])
    lst.append(globals()['A'+ str(L)])
    lst.append(parameters['W' +str(L)])
    lst.append(parameters['b' +str(L)])     
    cache = tuple(lst)
    
    return (globals()['A'+ str(L)]), cache


# ### Backward Propagation (linear -> relu ->linear ->relu.....->linear ->sigmoid) with Dropout

# In[25]:


# relu ->relu ->......->relu ->sigmoid
# cache has order ( Z1, DA1, A1, W1, b1, Z2, DA2, A2, W2, b2...., ZL, AL, WL, bL) ...last layer does not have mask DAL
def back_prop_with_dropout(X, Y, cache, keep_prob):  
    
    L = int((len(cache))/5) + 1
    m = X.shape[1]
    globals()['A'+str(0)] = X
    
    globals()['DA' +str(0)] = np.zeros((X.shape[0], X.shape[1]))
    
    #Accessing cache elements for hidden layers
    for i in range(1,L):   # retrieving the cache elements
        globals()['Z'+ str(i)]  =  cache[5*i - 5]
        globals()['DA' + str(i)] =  cache[5*i - 4]
        globals()['A' + str(i)] =  cache[5*i - 3]
        globals()['W' +str(i)]  =   cache[5*i - 2]
        globals()['b' +str(i)]  =   cache[5*i - 1]
    # Accessing cache elements for output layer "L"  
    globals()['Z'+ str(L)]  =  cache[5*L - 5]
    globals()['A' + str(L)] =  cache[5*L - 4]
    globals()['W' +str(L)]  =   cache[5*L - 3]
    globals()['b' +str(L)]  =   cache[5*L - 2]
    
    # back prop equations for last element , last layer uses sigmoid
    globals()['dZ' + str(L)] = (1/m)*((globals()['A' + str(L)]) - Y) # I have proved that, for softmax and sigmoid, this equation remains same
    
    # masking of last hidden layer with keep_prob
    #globals()['A'+ str(L-1)] = np.multiply(globals()['A' +str(L-1)], globals()['DA'+str(L-1)])
    #globals()['A'+ str(L-1)] = (globals()['A'+ str(L-1)])/keep_prob
    
    globals()['dW' + str(L)] = np.dot((globals()['dZ' + str(L)]), (globals()['A' + str(L -1)]).T)
    globals()['db' + str(L)] = np.sum((globals()['dZ' + str(L)]), axis = 1, keepdims = True)
    
    # masking of dA of last hidden layer
    globals()['dA' + str(L-1)]  = np.dot((globals()['W' +str(L)]).T , (globals()['dZ' + str(L)]))
    globals()['dA'+ str(L-1)] = np.multiply(globals()['dA' +str(L-1)], globals()['DA'+str(L-1)])
    globals()['dA'+ str(L-1)] = (globals()['dA'+ str(L-1)])/keep_prob
    
    # back prop equations for layers L-1, L-2, ....1
    for l in range(L-1, 0, -1):
        # transform A
        
        globals()['dZ' + str(l)] = np.multiply((globals()['dA' + str(l)]), np.int64((globals()['A' + str(l)])>0))
        
        #globals()['A'+ str(l-1)] = np.multiply(globals()['A' +str(l-1)], globals()['DA'+str(l-1)])
        #globals()['A'+ str(l-1)] = (globals()['A'+ str(l-1)])/keep_prob
        
        globals()['dW' + str(l)] = np.dot((globals()['dZ' + str(l)]), (globals()['A' + str(l -1)]).T)
        globals()['db' + str(l)] = np.sum((globals()['dZ' + str(l)]), axis = 1, keepdims = True)
        
        globals()['dA' + str(l-1)] = np.dot((globals()['W' +str(l)]).T , (globals()['dZ' + str(l)]))
        
        globals()['dA'+ str(l-1)] = np.multiply(globals()['dA' +str(l-1)], globals()['DA'+str(l-1)])
        globals()['dA'+ str(l-1)] = (globals()['dA'+ str(l-1)])/keep_prob

    # updating gradients in dictionary
    gradients = {}
    for i in range(1, L+1):
        gradients[('dW' +str(i))] = globals()['dW' +str(i)] 
        gradients[('db' +str(i))] = globals()['db' +str(i)]
    
    return gradients    
        


# ### Gradient Descent Update Rule

# In[26]:


def gradient_descent_update(parameters, gradients, learning_rate):
    L = int(len(parameters)/2)
    for l in range (1,L+1):
        parameters['W' +str(l)] = parameters['W' + str(l)] - learning_rate*gradients['dW' + str(l)]
        parameters['b' +str(l)] = parameters['b' + str(l)] - learning_rate*gradients['db' +str(l)]
    return parameters   


# ### Predict Accuracy with Basic Sigmoid ( For Test set only)

# In[27]:


def predict_accuracy(X, Y, parameters):
    m = Y.shape[1]
    AL,cache = forward_prop(X,parameters)
    AL_predicted = (AL > 0.5)
    accuracy = np.sum(np.float32(AL_predicted == Y))/m
    
    return accuracy
    
    
        


# ### Basic forward prop (no dropout) to calculate prediction on test set without dropout

# In[28]:


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


# ### Predict Accuracy with dropout Sigmoid ( For Training set)

# In[29]:


def predict_accuracy_with_dropout(X, Y, parameters, keep_prob):
    m = Y.shape[1]
    AL,cache = forward_prop_with_dropout(X,parameters, keep_prob)
    AL_predicted = (AL > 0.5)
    accuracy = np.sum(np.float32(AL_predicted == Y))/m
    
    return accuracy


# ### Sigmoid  Model with Dropout (with Gradient Descent, no mini batch)

# In[30]:


def dropout_model_sigmoid(X, Y, learning_rate, num_iteration, layers_dims, keep_prob):
    
    parameters = He_initialize_parameters(layers_dims)
    costs = []
    
    for i in range(0, num_iteration):
        AL, cache  = forward_prop_with_dropout(X, parameters, keep_prob)
        cost = compute_cost_sigmoid(AL, Y)
        if i%100==0:
             print('cost after ' +str(i) +' epochs is ' + str(cost))
        if i%10 ==0:
            costs.append(cost)
        grads  = back_prop_with_dropout(X, Y, cache, keep_prob)
        parameters = gradient_descent_update(parameters, grads, learning_rate)
     
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters  


