
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# #### One Hot Vector

# In[3]:


def one_hot(num_class, arr):
    one_hot = np.zeros((num_class, len(arr)))
    for i in range(0,len(arr)):
        temp = arr[i]
        one_hot[temp][i] = 1
    return one_hot    
        
        
            
    


# #### ReLu

# In[4]:


# ReLu
def relu(x):
    
   #  Arguments:
   #x -- A scalar or numpy array of any size.
    
    r = np.maximum(0,x)
    
    return r


# #### Softmax

# In[5]:


def softmax(Z):
    expZ = np.exp(Z)
    expZsum = np.sum(expZ, axis = 0,keepdims = True)
    softmax = expZ/expZsum
    return softmax
    
    


# #### Compute Cost for Softmax with regularization

# In[6]:


def regularized_compute_cost_softmax(AL, Y, parameters, lambd):
    m = Y.shape[1]
    L = int((1/2)*len(parameters))
    sumWsquared = 0.
    for l in range(1,L+1):
        sumWsquared = sumWsquared + np.sum(np.square(parameters['W'+str(l)]))
    J_Cross_Entropy = -(1/m)*np.nansum(np.nansum(np.multiply(Y, np.log(AL)), axis = 0,keepdims = True))
    J_Regularized_term = (lambd/(2*m))*sumWsquared
    J = J_Cross_Entropy + J_Regularized_term                                    
    return J


# In[1]:


#AL = softmax(np.random.randn(5,8))
#Y = one_hot(5,np.array([1,0,4,3,1,3,2,1]))
#lambd = 0.6
#layers_dims = [100,20,15,1]
#parameters=  He_initialize_parameters(layers_dims)
#cost = regularized_compute_cost_softmax(AL, Y, parameters, lambd)
#print(cost)


# #### He initialization

# In[9]:


def He_initialize_parameters(layers_dims):
    L = len(layers_dims)
    parameters = {}
    for k in range(1,L):
        parameters['W'+str(k)] = (np.random.randn(layers_dims[k],layers_dims[k-1]))*np.sqrt(2/layers_dims[k-1])
        parameters['b' +str(k)] = np.zeros((layers_dims[k],1))
    return parameters    


# ### Forward Propagation (   linear -> relu ->linear ->relu.....->linear ->sigmoid) same for Reguarized and non regularized version

# In[23]:


def forward_prop(X, parameters):   # linear -> relu ->linear ->relu.....->linear ->softmax
    
    L = int((1/2)*len(parameters))
    globals()['A'+ str(0)] = X
    for l in range(1, L):  # linear->relu upto L-1 layer
        globals()['Z'+ str(l)] = np.dot(parameters['W'+str(l)], (globals()['A'+ str(l-1)])) + parameters['b' +str(l)]
        globals()['A'+ str(l)] = relu((globals()['Z'+ str(l)]))
    
    globals()['Z'+ str(L)] = np.dot(parameters['W'+str(L)], (globals()['A'+ str(L-1)])) + parameters['b' +str(L)] 
    globals()['A'+ str(L)] = softmax((globals()['Z'+ str(L)]))                                
    # storing values in cache
    lst = []   # creating list and adding Z1, A1, W1, b1... then converting it into a tuple
    
    for l in range(1,L+1):                                  
        lst.append(globals()['Z'+ str(l)])
        lst.append(globals()['A'+ str(l)])                              
        lst.append(parameters['W' +str(l)])
        lst.append(parameters['b' +str(l)]) 
    cache = tuple(lst)
    
    return (globals()['A'+ str(L)]), cache


# ### Backward Propagation (linear -> relu ->linear ->relu.....->linear ->softmax) with Regularization

# In[21]:


# relu ->relu ->......->relu ->softmax    
# cache has order ( Z1, A1, W1, b1, Z2,, A2, W2, b2...)  Please note that dZapp( apparant dZ is different from real dZ)
def back_prop_with_regularization(X, Y, cache, lambd):   # dZapp means partial derivative of cross-entropy part
    L = int((len(cache))/4)                                 # of cost with respect to Z for any layer l. This is a tricky approach. Trick works to avoid mathematical complexity, think why?
    m = X.shape[1]
    globals()['A'+ str(0)] = X
    for i in range(1,L+1):   # retrieving the cache elements
        globals()['Z'+ str(i)]  =  cache[4*i - 4]
        globals()['A' + str(i)] =  cache[4*i - 3]
        globals()['W' +str(i)]  =   cache[4*i - 2]
        globals()['b' +str(i)]  =   cache[4*i - 1]
    # back prop equations for last element , last layer uses sigmoid
    globals()['dZapp' + str(L)] = (1/m)*((globals()['A' + str(L)]) - Y) # I have proved that, for softmax and sigmoid, this equation remains same
    globals()['dW' + str(L)] = np.dot((globals()['dZapp' + str(L)]), (globals()['A' + str(L -1)]).T) + (lambd/m)*(globals()['W' +str(L)])
    globals()['db' + str(L)] = np.sum((globals()['dZapp' + str(L)]), axis = 1, keepdims = True)
    globals()['dA' + str(L-1)]  = np.dot((globals()['W' +str(i)]).T , (globals()['dZapp' + str(L)]))
    
    # back prop equations for layers L-1, L-2, ....1
    for l in range(L-1, 0, -1):
        globals()['dZapp' + str(l)] = np.multiply((globals()['dA' + str(l)]), np.int64((globals()['A' + str(l)])>0))
        globals()['dW' + str(l)] = np.dot((globals()['dZapp' + str(l)]), (globals()['A' + str(l -1)]).T) + (lambd/m)*(globals()['W' +str(l)])
        globals()['db' + str(l)] = np.sum((globals()['dZapp' + str(l)]), axis = 1, keepdims = True)
        globals()['dA' + str(l-1)] = np.dot((globals()['W' +str(l)]).T , (globals()['dZapp' + str(l)]))

    # updating gradients in dictionary
    gradients = {}
    for i in range(1, L+1):
        gradients[('dW' +str(i))] = globals()['dW' +str(i)] 
        gradients[('db' +str(i))] = globals()['db' +str(i)]
    
    return gradients    
        


# #### Gradient Descent Update Rule

# In[25]:


def gradient_descent_update(parameters, gradients, learning_rate):
    L = int(len(parameters)/2)
    for l in range (1,L+1):
        parameters['W' +str(l)] = parameters['W' + str(l)] - learning_rate*gradients['dW' + str(l)]
        parameters['b' +str(l)] = parameters['b' + str(l)] - learning_rate*gradients['db' +str(l)]
    return parameters   


# #### Predict Accuracy with SoftMax (same for Regularization)

# In[26]:


def pred_accuracy_softmax(X,Y,parameters):
    AL, _ =       forward_prop(X, parameters)
    ALmax_index = np.argmax(AL, axis =0)
    Y_hat =       one_hot(AL.shape[0], ALmax_index )
    m =           Y.shape[1]
    predict_accuracy = (1/m)* np.sum(np.sum(np.multiply(Y, Y_hat), axis = 0))
    return predict_accuracy


# #### Basic Softmax Model (with Gradient Descent, no mini batch, no regularization)

# In[15]:


def regularized_model_softmax(X, Y, learning_rate, num_iterations, layers_dims, lambd): # simple model without minibatch, only gradient descent
    parameters = He_initialize_parameters(layers_dims)
    costs = []
    
    for i in range(0, num_iterations):
        AL, cache  = forward_prop(X, parameters)
        cost = regularized_compute_cost_softmax(AL, Y, parameters, lambd)
        if i%100==0:
            print('cost after ' +str(i) +' iterations is ' + str(cost))
        if i%10 ==0:
            costs.append(cost)
        grads  = back_prop_with_regularization(X, Y, cache, lambd)
        parameters = gradient_descent_update(parameters, grads, learning_rate)
        
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters    

    
    
    

