
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import random
import time


# #### One Hot Vector

# In[2]:


def one_hot(num_class, arr):
    one_hot = np.zeros((num_class, len(arr)))
    for i in range(0,len(arr)):
        temp = arr[i]
        one_hot[temp][i] = 1
    return one_hot    
        
        
            
    


# #### ReLu

# In[3]:


# ReLu
def relu(x):
    
   #  Arguments:
   #x -- A scalar or numpy array of any size.
    
    r = np.maximum(0,x)
    
    return r


# #### Softmax

# In[4]:


def softmax(Z):
    expZ = np.exp(Z)
    expZsum = np.sum(expZ, axis = 0,keepdims = True)
    softmax = expZ/expZsum
    return softmax
    
    


# #### Compute Cost for Softmax (same as for basic softmax)

# In[5]:


def compute_cost_softmax(AL, Y):
    m = Y.shape[1]
    cost = -(1/m)*np.nansum(np.nansum(np.multiply(Y, np.log(AL)), axis = 0,keepdims = True))
    return cost


# In[6]:


#AL = softmax(np.random.randn(5,8))
#Y = one_hot(5,np.array([1,0,4,3,1,3,2,1]))
#cost = compute_cost_softmax(Y,AL)
#print(cost)


# ### Compute Cost for sigmoid with Regularization

# In[ ]:


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


# #### He initialization

# In[7]:


def He_initialize_parameters(layers_dims):
    L = len(layers_dims)
    parameters = {}
    for k in range(1,L):
        parameters['W'+str(k)] = (np.random.randn(layers_dims[k],layers_dims[k-1]))*np.sqrt(2/layers_dims[k-1])
        parameters['b' +str(k)] = np.zeros((layers_dims[k],1))
    return parameters    


# ### Forward Propagation (   linear -> relu ->linear ->relu.....->linear ->softmax) same for Regularized and non regularized version

# In[ ]:


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


# ### Forward Propagation (   linear -> relu ->linear ->relu.....->linear ->softmax) with Dropout

# In[23]:


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
    globals()['A'+ str(L)] = softmax((globals()['Z'+ str(L)]))                                
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


# ### Backward Propagation (linear -> relu ->linear ->relu.....->linear ->softmax) with No Regularization and No Dropout

# In[ ]:


# relu ->relu ->......->relu ->softmax
# cache has order ( Z1, A1, W1, b1, Z2,, A2, W2, b2...)
def back_prop_without_regularization(X, Y, cache):  
    L = int((len(cache))/4)
    m = X.shape[1]
    globals()['A'+str(0)] = X
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
        


# ### Backward Propagation (linear -> relu ->linear ->relu.....->linear ->softmax) with Regularization

# In[ ]:


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
        


# ### Backward Propagation (linear -> relu ->linear ->relu.....->linear ->softmax) with Dropout

# In[24]:


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
        


# #### Predict Accuracy with SoftMax ( For Test set only)

# In[1]:


def pred_accuracy_softmax(X,Y,parameters):
    AL, _ =       forward_prop(X, parameters)
    ALmax_index = np.argmax(AL, axis =0)
    Y_hat =       one_hot(AL.shape[0], ALmax_index )
    m =           Y.shape[1]
    predict_accuracy = (1/m)* np.sum(np.sum(np.multiply(Y, Y_hat), axis = 0))
    return predict_accuracy


# ### Predict Accuracy with dropout Sigmoid ( For Training set)

# In[2]:


def predict_accuracy_with_dropout(X, Y, parameters, keep_prob):
    m = Y.shape[1]
    AL,cache = forward_prop_with_dropout(X,parameters, keep_prob)
    
    ALmax_index = np.argmax(AL, axis =0)
    Y_hat =       one_hot(AL.shape[0], ALmax_index )
    m =           Y.shape[1]
    predict_accuracy = (1/m)* np.sum(np.sum(np.multiply(Y, Y_hat), axis = 0))
    
    return predict_accuracy
    
  


# # OPTIMIZERS

# ### Gradient Descent( Batch Gradient Descent)

# In[3]:


def gradient_descent_update(parameters, gradients, learning_rate):
    L = int(len(parameters)/2)
    for l in range (1,L+1):
        parameters['W' +str(l)] = parameters['W' + str(l)] - learning_rate*gradients['dW' + str(l)]
        parameters['b' +str(l)] = parameters['b' + str(l)] - learning_rate*gradients['db' +str(l)]
    return parameters   


# ### Creating Random Mini Batches

# In[4]:


def random_minibatches(X_train, Y_train, minibatch_size):
    m = X_train.shape[1]
    # To shuffle X and Y train.
    #np.random.seed(0)
    K = list(np.random.permutation(m))  # k is an array, list() changes an array into a list. https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.permutation.html
    shuffled_X = X_train[:,K]
    shuffled_Y = Y_train[:,K]
    
    
    
    minibatches = []
    num_complete_minibatches = int(np.floor(m/minibatch_size))
    for k in range(0, num_complete_minibatches):
        minibatch_X = shuffled_X[:, k*minibatch_size:(k+1)*minibatch_size]
        minibatch_Y = shuffled_Y[:,k*minibatch_size:(k+1)*minibatch_size]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)
        
    # end case of mini batch
    if m % minibatch_size != 0:
        minibatch_X = shuffled_X[:,num_complete_minibatches*minibatch_size:m]
        minibatch_Y = shuffled_Y[:,num_complete_minibatches*minibatch_size:m]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)
    
    return minibatches    
    
    
    


# ### Momentum

# ### Initialize Velocities 

# In[5]:


def initialize_velocity(parameters):
    
    L = int((1/2)*len(parameters))
    V = {}
    for l in range(1, L+1):
        V['dW'+ str(l)] = np.zeros(((parameters['W'+str(l)]).shape))
        V['db'+ str(l)] = np.zeros(((parameters['b'+str(l)]).shape))
    
    return V     
        
        


# In[6]:


### Update Parameters with Momentum


# In[7]:


def momentum_update(parameters, gradients, learning_rate, V, beta1):
    
    L = int((1/2)*len(parameters))
    for l in range(1,L+1):
        V['dW'+str(l)] = beta1*V['dW'+str(l)] + (1-beta1)*gradients['dW'+str(l)]
        V['db'+str(l)] = beta1*V['db'+str(l)] + (1-beta1)*gradients['db'+str(l)]
        
        # parameters update
        parameters['W'+str(l)] = parameters['W'+str(l)] - learning_rate*V['dW'+str(l)]
        parameters['b'+str(l)] = parameters['b'+str(l)] - learning_rate*V['db'+str(l)]
        
    return parameters, V


# ### RMS Prop

# ### Initialize RMS Prop

# In[8]:


def initialize_rms_prop(parameters):
    
    L = int((1/2)*len(parameters))
    S = {}
    for l in range(1, L+1):
        S['dW'+ str(l)] = np.zeros(((parameters['W'+str(l)]).shape))
        S['db'+ str(l)] = np.zeros(((parameters['b'+str(l)]).shape))
    
    return S     
        
        


# ### Update Parameters with RMS Prop

# In[9]:


def rms_prop_update(parameters, gradients, learning_rate, S, beta2, epsilon = 1e-8):
    
    L = int((1/2)*len(parameters))
    
    for l in range(1, L+1):
        S['dW'+str(l)] = beta2*S['dW'+str(l)] + (1-beta2)*np.square(gradients['dW'+str(l)])
        S['db' + str(l)] = beta2*S['db' +str(l)] + (1-beta2)*np.square(gradients['db' +str(l)])
        
        #parameters update
        parameters['W'+str(l)] = parameters['W' +str(l)] - (learning_rate/(np.sqrt(S['dW'+str(l)])+ epsilon))*gradients['dW'+str(l)]
        parameters['b'+str(l)] = parameters['b' +str(l)] - (learning_rate/(np.sqrt(S['db'+str(l)])+ epsilon))*gradients['db'+str(l)]
        
    return parameters, S    


# ### ADAM

# ### Adam Initialization

# In[3]:


def initialize_adam(parameters):
    
    L = int((1/2)*len(parameters))
    V= {}
    S = {}
    for l in range(1, L+1):
        V['dW'+ str(l)] = np.zeros(((parameters['W'+str(l)]).shape))
        V['db'+ str(l)] = np.zeros(((parameters['b'+str(l)]).shape))
        S['dW'+ str(l)] = np.zeros(((parameters['W'+str(l)]).shape))
        S['db'+ str(l)] = np.zeros(((parameters['b'+str(l)]).shape))
        
    return V, S    


# ### Update Parameters with Adam

# In[4]:


def adam_update(parameters, gradients, learning_rate, V, S, t,  beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    
    L = int((1/2)*len(parameters))
    V_corrected = {}
    S_corrected = {}
    
    for l in range(1, L+1):
        
        V['dW'+str(l)] = beta1*V['dW'+str(l)] + (1-beta1)*gradients['dW'+str(l)]
        V_corrected['dW'+str(l)] = V['dW'+str(l)]/(1- (beta1)**t)
        
        V['db'+str(l)] = beta1*V['db'+str(l)] + (1-beta1)*gradients['db'+str(l)]
        V_corrected['db'+str(l)] = V['db'+str(l)]/(1- (beta1)**t)
        
        S['dW'+str(l)] = beta2*S['dW'+str(l)] + (1-beta2)*np.square(gradients['dW'+str(l)])
        S_corrected['dW'+str(l)] = S['dW'+str(l)]/(1- (beta2)**t)
        
        S['db'+str(l)] = beta2*S['db'+str(l)] + (1-beta2)*np.square(gradients['db'+str(l)])
        S_corrected['db'+str(l)] = S['db'+str(l)]/(1- (beta2)**t)
        
        #parameters update
        parameters['W'+str(l)] = parameters['W'+str(l)] - (learning_rate/(np.sqrt(S_corrected['dW'+str(l)]) + epsilon))*V_corrected['dW'+str(l)]
        parameters['b'+str(l)] = parameters['b'+str(l)] - (learning_rate/(np.sqrt(S_corrected['db'+str(l)]) + epsilon))*V_corrected['db'+str(l)]
    
    return parameters, V, S


# ## Softmax  Model with Regularization_OR_Dropout (with different Optimizers)

# In[5]:


# this definition allows only to use regularized or dropout at a time although it is possible to use both at the same time.
def regularized_or_dropout_optimizers_model_softmax(X, Y, layers_dims, minibatch_size, lambd, keep_prob, optimizer, num_epochs, learning_rate, beta1, beta2, epsilon):
    
    #initialize parameters
    parameters = He_initialize_parameters(layers_dims)
    costs = []
    
    # initialize adam counter
    t = 0
    # initialize optimizers
    if optimizer == 'gd':
        pass
    elif optimizer == 'momentum':
        V = initialize_velocity(parameters)
    elif optimizer == 'rms_prop':
        S = initialize_rms_prop(parameters) 
    elif optimizer == 'adam':
        V, S = initialize_adam(parameters)
    
    
    if lambd != 0:
        for i in range(0, num_epochs):
            minibatches = random_minibatches(X, Y, minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                AL, cache  = forward_prop(minibatch_X, parameters)
                grads  = back_prop_with_regularization(minibatch_X, minibatch_Y, cache, lambd)
                
                if optimizer  == 'gd':
                    parameters = gradient_descent_update(parameters, grads, learning_rate)
                elif optimizer == 'momentum':
                    parameters, V = momentum_update(parameters, grads, learning_rate,V, beta1)
                elif optimizer == 'rms_prop':
                    parameters, S = rms_prop_update(parameters, grads, learning_rate, S, beta2, epsilon)
                elif optimizer == 'adam':
                    t = t+1  # increment adam counter... I have to know why to do this
                    parameters, V, S == adam_update(parameters, grads, learning_rate, V, S, t, beta1, beta2, epsilon)
                
                cost = regularized_compute_cost_softmax(AL, minibatch_Y, parameters, lambd)
        
            if i%100==0:
                print('cost after ' +str(i) +' epochs is ' + str(cost))
            if i%10 ==0:
                costs.append(cost)
        
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
    elif keep_prob !=0:
        for i in range(0, num_epochs):
            minibatches = random_minibatches(X, Y, minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                AL, cache  = forward_prop_with_dropout(minibatch_X, parameters, keep_prob)
                grads  = back_prop_with_dropout(minibatch_X, minibatch_Y, cache, keep_prob)
                
                if optimizer  == 'gd':
                    parameters = gradient_descent_update(parameters, grads, learning_rate)
                elif optimizer == 'momentum':
                    parameters, V = momentum_update(parameters, grads, learning_rate,V, beta1)
                elif optimizer == 'rms_prop':
                    parameters, S = rms_prop_update(parameters, grads, learning_rate, S, beta2, epsilon)
                elif optimizer == 'adam':
                    t = t+1  # increment adam counter... I have to know why to do this
                    parameters, V, S == adam_update(parameters, grads, learning_rate, V, S, t, beta1, beta2, epsilon)
                
                cost = compute_cost_softmax(AL, minibatch_Y)
        
            if i%100==0:
                print('cost after ' +str(i) +' epochs is ' + str(cost))
            if i%10 ==0:
                costs.append(cost)
        
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
    elif (lambd== 0) and (keep_prob ==0):
        for i in range(0, num_epochs):
            minibatches = random_minibatches(X, Y, minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                AL, cache  = forward_prop(minibatch_X, parameters)
                grads  = back_prop_without_regularization(minibatch_X, minibatch_Y, cache)
                
                if optimizer  == 'gd':
                    parameters = gradient_descent_update(parameters, grads, learning_rate)
                elif optimizer == 'momentum':
                    parameters, V = momentum_update(parameters, grads, learning_rate,V, beta1)
                elif optimizer == 'rms_prop':
                    parameters, S = rms_prop_update(parameters, grads, learning_rate, S, beta2, epsilon)
                elif optimizer == 'adam':
                    t = t+1  # increment adam counter... I have to know why to do this
                    parameters, V, S == adam_update(parameters, grads, learning_rate, V, S, t, beta1, beta2, epsilon)
                
                cost = compute_cost_softmax(AL, minibatch_Y)
            
            if i%100==0:
                print('cost after ' +str(i) +' epochs is ' + str(cost))
            if i%10 ==0:
                costs.append(cost) 
       
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        
    return parameters  



# ## Automated Hyperparameters Tuning

# ### Defining Sets Of Random Hyperparameters

# #### define set of random layers_dims (random number of layers and neurons per layer)

# In[6]:


def random_layers_dims(X_train, Y_train):
    num_hidden_layers = np.random.randint(1,5)
    layers_dims_array = np.random.randint(10,20, size = num_hidden_layers)
    layers_dims = layers_dims_array.tolist()
    layers_dims.insert(0, X_train.shape[0])
    layers_dims.append(Y_train.shape[0])
    return layers_dims


# #### define random alphax

# In[7]:


def random_alpha():
    alphaindex = -4*np.random.rand()
    alpha = 10**alphaindex
    return alpha


# #### define set of random minibatches

# In[8]:


def random_minibatch_size(X_train):
    minibatch_size_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, X_train.shape[1]]
    minibatch_size = random.choice(minibatch_size_list)
    return minibatch_size


# #### define set of random optimizer

# In[9]:


def random_optimizer():  
    optimizers_list = ['gd', 'momentum', 'rms_prop', 'adam']
    optimizer = random.choice(optimizers_list)
    return optimizer
    


# #### define random keep_prob

# In[10]:


def random_keep_prob():
    keep_prob = np.random.rand()
    return keep_prob

    


# #### define random lambd(regularization constant)

# In[11]:


def random_lambd():
    lambd = 2*np.random.rand()
    return lambd


# ### Automated Hyperparameter Tuning Model

# #### No Regularization, no dropout

# In[12]:


def automated_hyperparameters_tuning_no_regularization_no_dropout(X_train, Y_train, X_test, Y_test):
    
    train_accuracy = 0
    test_accuracy = 0
    lst_param_train_test_accuracy = []
    while(test_accuracy < 0.94):
        layers_dims = random_layers_dims(X_train, Y_train)
        minibatch_size = random_minibatch_size(X_train)
        optimizer = random_optimizer()
        learning_rate = random_alpha()
        tic = time.time()  # to calculate time of training
        parameters = regularized_or_dropout_optimizers_model_softmax(X_train, Y_train, layers_dims = layers_dims, minibatch_size = minibatch_size, lambd = 0, keep_prob = 0, optimizer = optimizer,  num_epochs = 1, learning_rate = learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
        tac = time.time()
        training_time = tac-tic
        train_accuracy = pred_accuracy_softmax(X_train, Y_train, parameters)
        test_accuracy = pred_accuracy_softmax(X_test, Y_test, parameters)

        
        tpl_param_accuracy = (parameters, train_accuracy, test_accuracy, training_time)
        lst_param_train_test_accuracy.append(tpl_param_accuracy)
        parameters_for_max_test_accuracy, max_train_accuracy, max_test_accuracy, max_training_time = max(lst_param_train_test_accuracy,key=lambda item:item[2])
        
        hyperparameters_for_max_test_accuracy = (layers_dims, minibatch_size, optimizer, learning_rate)
        
        print('Train accuracy for current model is: '+str(train_accuracy))
        print('Test accuracy for current model is: '+str(test_accuracy))
        print('Time taken in training current model is :' + str(training_time) +' seconds')
        
        print('Maximum Test accuracy until now using different set of hyperparameters is ' +str(max_test_accuracy))
        print('Train accuracy corresponding to maximum test accuracy  ' +str(max_train_accuracy))
        print('Time taken in training model of maximum test accuracy is :' + str(max_training_time) +' seconds')
        
    return parameters_for_max_test_accuracy, hyperparameters_for_max_test_accuracy
    
                                                                    
                                                                    
    


# #### Regularization version

# In[13]:


def automated_hyperparameters_tuning_regularized_version(X_train, Y_train, X_test, Y_test):
    
    train_accuracy = 0
    test_accuracy = 0
    lst_param_train_test_accuracy = []
    while(test_accuracy < 0.94):
        layers_dims = random_layers_dims(X_train, Y_train)
        minibatch_size = random_minibatch_size(X_train)
        lambd = random_lambd()
        optimizer = random_optimizer()
        learning_rate = random_alpha()
        tic = time.time()
        parameters = regularized_or_dropout_optimizers_model_softmax(X_train, Y_train, layers_dims = layers_dims, minibatch_size = minibatch_size, lambd = lambd, keep_prob = 0, optimizer = optimizer,  num_epochs = 1, learning_rate = learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
        tac = time.time()
        training_time = tac-tic
        train_accuracy = pred_accuracy_softmax(X_train, Y_train, parameters)
        test_accuracy = pred_accuracy_softmax(X_test, Y_test, parameters)

        
        tpl_param_accuracy = (parameters, train_accuracy, test_accuracy, training_time)
        lst_param_train_test_accuracy.append(tpl_param_accuracy)
        parameters_for_max_test_accuracy, max_train_accuracy, max_test_accuracy, max_training_time = max(lst_param_train_test_accuracy,key=lambda item:item[2])
        
        hyperparameters_for_max_test_accuracy = (layers_dims, minibatch_size, lambd, optimizer, learning_rate)
        
        print('Train accuracy for current model is: '+str(train_accuracy))
        print('Test accuracy for current model is: '+str(test_accuracy))
        print('Time taken in training current model is :' + str(training_time) +' seconds')
        
        print('Maximum Test accuracy until now using different set of hyperparameters is ' +str(max_test_accuracy))
        print('Train accuracy corresponding to maximum test accuracy  ' +str(max_train_accuracy))
        print('Time taken in training model of maximum test accuracy is :' + str(max_training_time) +' seconds')
        
    return parameters_for_max_test_accuracy, hyperparameters_for_max_test_accuracy
    
                                                                    
                                                                    
    


# #### Dropout version

# In[14]:


def automated_hyperparameters_tuning_dropout_version(X_train, Y_train, X_test, Y_test):
    
    train_accuracy = 0
    test_accuracy = 0
    lst_param_train_test_accuracy = []
    while(test_accuracy < 0.94):
        layers_dims = random_layers_dims(X_train, Y_train)
        minibatch_size = random_minibatch_size(X_train)
        keep_prob = random_keep_prob()
        optimizer = random_optimizer()
        learning_rate = random_alpha()
        tic = time.time()
        parameters = regularized_or_dropout_optimizers_model_softmax(X_train, Y_train, layers_dims = layers_dims, minibatch_size = minibatch_size, lambd = 0, keep_prob = keep_prob, optimizer = optimizer,  num_epochs = 1, learning_rate = learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
        tac = time.time()
        training_time = tac - tic
        train_accuracy = predict_accuracy_with_dropout(X_train, Y_train, parameters, keep_prob)
        test_accuracy = pred_accuracy_softmax(X_test, Y_test, parameters)

        
        tpl_param_accuracy = (parameters, train_accuracy, test_accuracy,training_time)
        lst_param_train_test_accuracy.append(tpl_param_accuracy)
        parameters_for_max_test_accuracy, max_train_accuracy, max_test_accuracy, max_training_time = max(lst_param_train_test_accuracy,key=lambda item:item[2])
        
        hyperparameters_for_max_test_accuracy = (layers_dims, minibatch_size, keep_prob, optimizer, learning_rate)
        
        print('Train accuracy for current model is: '+str(train_accuracy))
        print('Test accuracy for current model is: '+str(test_accuracy))
        print('Time taken in training current model is :' + str(training_time) +' seconds')
        
        print('Maximum Test accuracy until now using different set of hyperparameters is ' +str(max_test_accuracy))
        print('Train accuracy corresponding to maximum test accuracy  ' +str(max_train_accuracy))
        print('Time taken in training model of maximum test accuracy is :' + str(max_training_time) +' seconds')
        
    return parameters_for_max_test_accuracy, hyperparameters_for_max_test_accuracy
    
                                                                    
                                                                    
    

