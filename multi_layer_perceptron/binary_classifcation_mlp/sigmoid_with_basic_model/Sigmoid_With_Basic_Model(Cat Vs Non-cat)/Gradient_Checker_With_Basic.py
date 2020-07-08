
# coding: utf-8

# ## Gradient Checker (for Basic Sigmoid no regularization)

# In[2]:


import numpy as np


# #### ReLu (needed for forward prop in grad checker)

# In[3]:


# ReLu
def relu(x):
    
   #  Arguments:
   #x -- A scalar or numpy array of any size.
    
    r = np.maximum(0,x)
    
    return r


# #### sigmoid( needed for forward prop in grad checker)

# In[4]:


# Sigmoid
def sigmoid(x):
    
    #  Arguments:
    #x -- A scalar or numpy array of any size.
    
    s = 1/(1+np.exp(-x))
    return s


# #### compute cost( needed for grad checker)

# In[5]:


def compute_cost_sigmoid(AL, Y):
    m = Y.shape[1]
    J = -(1/m)*np.sum((np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL))))
    return J


# #### forward prop needed for grad checker

# In[6]:


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


# #### Dictionary(Parameters) To Vector

# In[7]:


def dictionary_to_vector(parameters):
    L = len(parameters)//2
    
    for l in range(1,L+1):
        globals()['W_vec' +str(l)] = np.reshape(parameters['W' + str(l)], (-1,1))  
        globals()['b_vec' +str(l)] = np.reshape(parameters['b' + str(l)], (-1,1)) 
        new_vec = np.concatenate((globals()['W_vec'+ str(l)], globals()['b_vec'+ str(l)]), axis = 0 )
        if l==1 :
            theta = new_vec
        else:    
            theta = np.concatenate((theta, new_vec), axis = 0)
    
    # returning parameter dimensions as list to get back from vector to parameters
    param_dims = []
    for l in range (1,L+1):
        param_dims.append(parameters['W' + str(l)].shape)
        param_dims.append((parameters['W' + str(l)].shape[0])*(parameters['W' + str(l)].shape[1]))
        param_dims.append(parameters['b' + str(l)].shape)
        param_dims.append((parameters['b' + str(l)].shape[0])*(parameters['b' + str(l)].shape[1]))
    
    return theta, param_dims


# #### Vector to Dictionary (Parameters)

# In[8]:


## Vector_to_dictionary

def vector_to_dictionary(theta, param_dims):
    parameters = {}
    L = len(param_dims)//4
    
     
    for l in range(1, L+1):
        
         
        # to get parameter elements, indexW refers to indices of W's of param_dims
        indexW = 1 + (l-1)*4   # n th element of an AP, to get W1, W2 etc
        if indexW == 1:
            parameters['W'+str(l)] = theta[0:param_dims[indexW]].reshape((param_dims[4*(l-1)]))
        
        else:
            vec_index = 0
            for x in range(1,indexW , 2):   # adding before the current index of param_dims
                vec_index = vec_index + param_dims[x]
            parameters['W'+str(l)] = theta[vec_index: vec_index + param_dims[indexW]].reshape((param_dims[4*(l-1)]))
        # to get parameter elements, indexb refers to indices of b's of param_dims
        
        indexb = 3 + (l-1)*4
        if indexb == 3:
            parameters['b'+str(l)] = theta[param_dims[indexW]: param_dims[indexW] + param_dims[indexb]].reshape((param_dims[2+ 4*(l-1)])) 
        else:
            vec_index = 0  
            for x in range(1, indexb, 2):
                vec_index = vec_index + param_dims[x]
            parameters['b'+str(l)] = theta[vec_index: vec_index + param_dims[indexb]].reshape((param_dims[2+ 4*(l-1)])) 
    
    return  parameters                                   


# #### Gradients (Dictionary) to Vector

# In[9]:


## Gradients_to_vectors
def gradients_to_vector(gradients):
    L = np.int(len(gradients)/2)

    
    for l in range(1,L+1):
        globals()['dW_vec' +str(l)] = np.reshape(gradients['dW' + str(l)], (-1,1))  
        globals()['db_vec' +str(l)] = np.reshape(gradients['db' + str(l)], (-1,1)) 
        new_vec = np.concatenate((globals()['dW_vec'+ str(l)], globals()['db_vec'+ str(l)]), axis = 0 )
        if l==1 :
            theta_grad = new_vec
        else:    
            theta_grad = np.concatenate((theta_grad, new_vec), axis = 0)
    
    return theta_grad


# #### Gradient Checker

# In[10]:


def gradient_check(X,Y, parameters, gradients, epsilon = 1e-7):
    theta, param_dims = dictionary_to_vector(parameters)
    num_parameters = theta.shape[0]
    Jplus = np.zeros((num_parameters,1))
    Jminus = np.zeros((num_parameters,1))
    grad_approx = np.zeros((num_parameters,1))
    grad = gradients_to_vector(gradients)
    
    for i in range (num_parameters):
        
        thetaplus = np.copy(theta)
        thetaplus[i][0] = thetaplus[i][0] + epsilon
        parameters_new = vector_to_dictionary(thetaplus, param_dims)
        ALplus, _ = forward_prop(X, parameters_new)
        Jplus[i] = compute_cost_sigmoid(ALplus, Y)
        
        thetaminus = np.copy(theta)
        thetaminus[i][0] = thetaminus[i][0] - epsilon
        ALminus, _ = forward_prop(X, vector_to_dictionary(thetaminus, param_dims))
        Jminus[i] = compute_cost_sigmoid(ALminus, Y)
        
        grad_approx[i] = (Jplus[i] - Jminus[i])/(2*epsilon)
        
        print('gradient using grad checker is' +str(grad_approx[i]))
        print('gradient using back prop is' +str(grad[i]))
    
    numerator = np.linalg.norm(grad - grad_approx)
   
        
    denominator = np.linalg.norm(grad) + np.linalg.norm(grad_approx)
     
        
    difference = numerator/denominator
        
    if difference < 2e-7:
        print('Your back prop is working absolutely fine. Difference is ' + str(difference))
    if difference > 2e-7:
        print('check your back prop..Error in back prop. Differnece is ' + str(difference))

    return difference    
        

