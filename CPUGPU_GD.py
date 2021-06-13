import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from random import randint
from random import random
from random import seed
import math
from sklearn.utils import shuffle

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule
import time
from concurrent.futures import ProcessPoolExecutor
from keras.datasets import mnist

train_data,test_data=mnist.load_data()

train_data_mean = np.mean(train_data[0])
train_data_stdev = np.std(train_data[0])
train_data = ((train_data[0] - train_data_mean) / train_data_stdev, train_data[1])
test_data = ((test_data[0] - train_data_mean) / train_data_stdev, test_data[1])
train_X, trainY = train_data
test_X, testY = test_data


train_X, trainY = shuffle(train_X, trainY)
test_X, testY = shuffle(test_X, testY)


print('Train: X=%s, y=%s' %(train_X.shape,trainY.shape))
print('Test: X=%s, y=%s' %(test_X.shape,testY.shape))

trainX=[train_X[i].flatten() for i in range(len(train_X))]
testX=[test_X[i].flatten() for i in range(len(test_X))]

train_samples=len(trainX)
test_samples=len(testX)
xlen=len(trainX[0])
print("Input vector size:",xlen)
print("\n")

mod = SourceModule("""
    #include "cmath"
    __global__ void compute_fw( int m, int n, float *a, float *b, float *bias ,float *c)
    {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
      	if (id < m)
	    {
	    	c[id] = bias[id];
	      	for (int k = 0; k < n; k++)
            {
		      	c[id] += a[id * n + k] * b[k];
            }
	    }
    }
    __global__ void relu_ac( int sz, float* s, float* d)
    {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if(id < sz)
        {
            if(s[id]<0)
            d[id]=0;
            else
            d[id]=s[id];
        }
    }
    __global__ void grad_mul(int d1,int d2, float* a, float* b,float* c)
    {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if(id<d1*d2)
        {
            int row = id/d2;
            int col = id%d2;
            c[id] = a[row]*b[col];
        }
    }
    __global__ void grad_relu(int d1, float* a, float* b)
    {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if(id<d1)
        {
            if(a[id]<0)
            b[id] = 0;
            else
            b[id] = 1;
        }
    }
     __global__ void ele_grad(int d1, float* a, float* b, float* c)
     {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if(id<d1)
        {
            c[id] = a[id]*b[id];
        }
     }
     __global__ void grad_wt(int m, int n, float* a, float* b, float* c)
     {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if(id<n)
        {
            c[id] = 0;
            for(int i=0;i<m;i++)
            {
                c[id] += a[n*i+id] * b[i]; 
            }
        }
     }
     __global__ void update(float sf,int n, float* a,float* b)
     {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if(id<n)
        {
            b[id] -= sf*a[id];
        }
     }
     __global__ void add(int n,float* a,float* b)
     {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if(id<n)
        {
          a[id]=a[id]+b[id];
        }
     }
     __global__ void reset(int n, float* a)
     {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if(id<n)
        {
            a[id] = 0;
        }
     }
     __global__ void lookahead_c(int n,float gamma, float* a,float* b,float* c)
     {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if(id<n)
        {
            c[id] = a[id] - (gamma*b[id]);
        }
     }
     __global__ void update_1(int n,float gamma, float eta, float* a,float* b, float* c)
     {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if(id<n)
        {
           a[id] =  gamma*a[id] + eta*b[id];
           c[id] = c[id] - a[id];
        }
     }
     __global__ void update_2(int n,float beta, float* a,float* b)
     {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if(id<n)
        {
           a[id] = beta*a[id] + (1-beta)*b[id]*b[id];
        }
     }
     __global__ void update_3(int n,float epsilon,float eta,float* a,float* b,float* c)
     {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if(id<n)
        {
          c[id] -= (eta/sqrt(a[id] + epsilon)) * b[id];
        }
     }
     __global__ void update_4(int n,float beta, float* a,float* b)
     {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if(id<n)
        {
           a[id] = beta*a[id] + (1-beta)*b[id];
        }
     }
     __global__ void update_5(int n,float epsilon,float eta,float beta1,float beta2,int t,float* a,float* b,float* c)
     {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if(id<n)
        {
          float ac = a[id]/(1-powf(beta2,t));
          float bc = b[id]/(1-powf(beta1,t));
          c[id] -= (eta/sqrt(ac + epsilon) * bc);
        }
     }
     __global__ void update_6(int n,float epsilon,float eta,float beta1,float beta2,int t,float* a,float* b,float* g,float* c)
     {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if(id<n)
        {
          float ac = a[id]/(1-powf(beta2,t));
          float bc = b[id]/(1-powf(beta1,t));
          float md = beta1*bc + (1-beta1)*g[id];
          c[id] -= (eta/sqrt(ac + epsilon) * md);
        }
     }
""")


def initialize_weights(n_input, n_hidden_layer, n_output,neurons_hl):
    W = list()
    W.append([[np.random.normal(0,1/n_input) for j in range(n_input)]for i in range(neurons_hl)])
    for i in range(n_hidden_layer-1):
        W.append([[np.random.normal(0,1/neurons_hl) for j in range(neurons_hl)]for i in range(neurons_hl)])
    W.append([[np.random.normal(0,1/neurons_hl) for j in range(neurons_hl)]for i in range(n_output)])
    return W

def initialize_weights_zero(n_input, n_hidden_layer, n_output,neurons_hl):
    W = list()
    W.append([[0.0 for j in range(n_input)]for i in range(neurons_hl)])
    for i in range(n_hidden_layer-1):
        W.append([[0.0 for j in range(neurons_hl)]for i in range(neurons_hl)])
    W.append([[0.0 for j in range(neurons_hl)]for i in range(n_output)])
    return W

def initialize_weights_zero_gpu(n_input, n_hidden_layer, n_output,neurons_hl):
    W = initialize_weights_zero(n_input, n_hidden_layer, n_output,neurons_hl)
    W_ptr = []
    for weight in W:
        a = np.array(weight)
        a = a.astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        W_ptr.append(a_gpu)
    return W_ptr

def initialize_weights_gpu(n_input, n_hidden_layer, n_output,neurons_hl):
    W = initialize_weights(n_input, n_hidden_layer, n_output,neurons_hl)
    W_ptr = []
    for weight in W:
        a = np.array(weight)
        a = a.astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        W_ptr.append(a_gpu)
    return W_ptr

def initialize_bias(n_input, n_hidden_layer, n_output,neurons_hl):
    B=list()
    for i in range(n_hidden_layer):
        B.append([0 for i in range(neurons_hl)])
    B.append([0 for i in range(n_output)])
    return B

def initialize_bias_gpu(n_input, n_hidden_layer, n_output,neurons_hl):
    B = initialize_bias(n_input, n_hidden_layer, n_output,neurons_hl)
    B_ptr = []
    for bias in B:
        a = np.array(bias)
        a = a.astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        B_ptr.append(a_gpu)
    return B_ptr



def relu(b):
  #a=np.array(b,dtype=np.float128)
  return np.maximum(b,0)

def diff_relu(a):
  res=list()
  for el in a:
    if el<0:
      res.append(0)
    else:
      res.append(1)
  return res

def softmax(a):
  z = a - max(a)
  h=list()
  sum=0
  for el in z:
    sum+=np.exp(el)
  for el in z:
    h.append(np.exp(el)/sum)
  return h

def feed_forward_gpu(input,W,B,L):
    H= []
    A= []
    input = input.astype(np.float32)
    ip_g = gpuarray.to_gpu(np.array(input))
    H.append(ip_g)
    A.append([0])
    for i in range(1,L):
        b=B[i-1].shape
        w=W[i-1].shape
        a_gpu = gpuarray.empty(b, np.float32)
        h_gpu = gpuarray.empty(b, np.float32)
        bs=32
        gs=math.ceil(float(float(b[0])/bs))
        func = mod.get_function("compute_fw")
        func(np.int32(w[0]),np.int32(w[1]), W[i-1],H[i-1],B[i-1],a_gpu, block=(bs, 1, 1), grid=(gs, 1), shared=0)
        A.append(a_gpu)
        func = mod.get_function("relu_ac")
        func(np.int32(b),a_gpu, h_gpu, block=(bs, 1, 1), grid=(gs, 1), shared=0)
        H.append(h_gpu)
    b=B[L-1].shape
    w=W[L-1].shape
    a_gpu = gpuarray.empty(b, np.float32)
    bs=32
    gs=math.ceil(float(float(b[0])/bs))
    func = mod.get_function("compute_fw")
    func(np.int32(w[0]),np.int32(w[1]), W[L-1], H[L-1], B[L-1],a_gpu, block=(bs, 1, 1), grid=(gs, 1), shared=0)
    A.append(a_gpu)
    a_cpu=a_gpu.get()
    hL=softmax(a_cpu)
    h_gpu=gpuarray.to_gpu(np.array(hL))
    H.append(h_gpu)
    return H,A,hL

def feed_forward(input,W,B,L):
    H=list()
    A=list()
    H.append(input)
    A.append([0])
    for i in range(1,L):
        a=B[i-1]+np.matmul(W[i-1],H[i-1])
        A.append(a)
        H.append(relu(a))
    a=B[L-1]+np.matmul(W[L-1],H[L-1])
    A.append(a)
    hL=softmax(a)
    H.append(hL)
    return H,A,hL


def back_propogation_gpu(H,A,y_hat,label,W,L,K):
    W_grad=list()
    B_grad=list()
    one_hot_y=np.zeros(K)
    one_hot_y[label]+=1
    ak_gradc = y_hat-one_hot_y
    ak_gradc = np.array(ak_gradc)
    ak_gradc = ak_gradc.astype(np.float32)
    ak_grad = gpuarray.to_gpu(ak_gradc)
    for k in range(L,0,-1):
        d1=ak_grad.shape[0]
        d2=H[k-1].shape[0]
        w_grad=gpuarray.empty((d1,d2),np.float32)
        bs=32
        gs=math.ceil(float(float(d1*d2)/bs))
        func = mod.get_function("grad_mul")
        func(np.int32(d1),np.int32(d2), ak_grad, H[k-1], w_grad, block=(bs, 1, 1), grid=(gs, 1), shared=0)
        #w_grad=np.matmul(np.matrix(ak_grad).T,np.matrix(H[k-1]))
        W_grad.append(w_grad)
        B_grad.append(ak_grad)
        if k != 1:
            d3 = W[k-1].shape[1]
            h_grad=gpuarray.empty((d3,1),np.float32)
            bs=32
            gs=math.ceil(float(float(d3)/bs))
            func = mod.get_function("grad_wt")
            func(np.int32(d1),np.int32(d3), W[k-1], ak_grad,h_grad, block=(bs, 1, 1), grid=(gs, 1), shared=0)
            ak_grad=gpuarray.empty((d3,1),np.float32)
            diff=gpuarray.empty((d3,1),np.float32)
            func = mod.get_function("grad_relu")
            func(np.int32(d3), A[k-1] ,diff, block=(bs, 1, 1), grid=(gs, 1), shared=0)
            func = mod.get_function("ele_grad")
            func(np.int32(d3), h_grad,diff,ak_grad, block=(bs, 1, 1), grid=(gs, 1), shared=0)
            #h_grad=np.matmul(np.transpose(W[k-1]),ak_grad)
            #ak_grad=np.multiply(h_grad,diff_relu(A[k-1]))
    return W_grad,B_grad


def back_propogation(H,A,y_hat,label,W,L,K):
    W_grad=list()
    B_grad=list()
    one_hot_y=np.zeros(K)
    one_hot_y[label]+=1
    ak_grad = y_hat-one_hot_y
    for k in range(L,0,-1):
        w_grad=np.matmul(np.matrix(ak_grad).T,np.matrix(H[k-1]))
        W_grad.append(w_grad)
        B_grad.append(ak_grad)
        if k != 1:
            h_grad=np.matmul(np.transpose(W[k-1]),ak_grad)
            ak_grad=np.multiply(h_grad,diff_relu(A[k-1]))
    return W_grad,B_grad


def stochastic_gd_gpu():
    e=0
    epoch=1
    error=0.0
    W= initialize_weights_gpu(xlen,L-1,K,N)
    B= initialize_bias_gpu(xlen,L-1,K,N)
    loss=list()
    while (e<epoch):
        for i in range(train_samples):
            H,A,y_hat=feed_forward_gpu(trainX[i],W,B,L)
            loss.append(-math.log(y_hat[trainY[i]]))
            W_grad,B_grad=back_propogation_gpu(H,A,y_hat,trainY[i],W,L,K)
            W_grad=W_grad[::-1]
            B_grad=B_grad[::-1]
            for l in range(L):
                a1 = W_grad[l].shape[0]
                a2 = W_grad[l].shape[1]
                b1 = B_grad[l].shape[0]
                bs=32
                gs=math.ceil(float(float(a1*a2)/bs))
                func = mod.get_function("update")
                func(np.float32(eta),np.int32(a1*a2), W_grad[l], W[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
                gs=math.ceil(float(float(b1)/bs))
                func(np.float32(eta),np.int32(b1), B_grad[l], B[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)  
        e=e+1
        print('Epoch done')
    return W,B,loss

def stochastic_gd():
    e=0
    epoch=1
    error=0.0
    W= initialize_weights(xlen,L-1,K,N)
    B= initialize_bias(xlen,L-1,K,N)
    loss=list()
    while (e<epoch):
        begin=time.time()
        for i in range(train_samples):
            H,A,y_hat=feed_forward(trainX[i],W,B,L)
            loss.append(-math.log(y_hat[trainY[i]]))
            W_grad,B_grad=back_propogation(H,A,y_hat,trainY[i],W,L,K)
            W_grad=W_grad[::-1]
            B_grad=B_grad[::-1]
            for l in range(L):
                W[l]=(np.matrix(W[l])-np.multiply(eta,W_grad[l])).tolist()
                B[l]=np.subtract(B[l],np.multiply(eta,B_grad[l])).tolist()       
        end=time.time()
        e=e+1
        print('Epoch done')
    return W,B,loss

def concurrent_batch(arg):
    input=arg[0]
    output=arg[1]
    W,B,L,K = arg[2],arg[3],arg[4],arg[5]
    error=0
    H,A,y_hat=feed_forward_gpu(input,W,B,L)
    ei=y_hat[output]
    if ei!=0:
     error = -math.log(ei+1e-7)
    W_grad,B_grad=back_propogation_gpu(H,A,y_hat,output,W,L,K)
    W_grad=W_grad[::-1]
    B_grad=B_grad[::-1]
    return W_grad,B_grad,error


def momentum_gpu():
    epoch=1
    batch_size=32
    t,e= 0,0
    W= initialize_weights_gpu(xlen,L-1,K,N)
    B= initialize_bias_gpu(xlen,L-1,K,N)
    points= 0
    error= 0.0
    gamma= 0.9
    loss=list()
    wgrad= initialize_weights_zero_gpu(xlen,L-1,K,N)
    bgrad= initialize_bias_gpu(xlen,L-1,K,N)
    uw= initialize_weights_zero_gpu(xlen,L-1,K,N)
    ub= initialize_bias_gpu(xlen,L-1,K,N)
    while (e<epoch):
        for i in range(train_samples):
            H,A,y_hat=feed_forward_gpu(trainX[i],W,B,L)
            ei=y_hat[trainY[i]]
            if ei!=0:
                error += -math.log(ei+1e-7)
            W_grad,B_grad=back_propogation_gpu(H,A,y_hat,trainY[i],W,L,K)
            W_grad=W_grad[::-1]
            B_grad=B_grad[::-1]
            for l in range(L):
                (f1,f2) = wgrad[l].shape
                b1 = bgrad[l].shape[0]
                bs=32
                func = mod.get_function("add")
                gs=math.ceil(float(float(f1*f2)/bs))
                func(np.int32(f1*f2),wgrad[l],W_grad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
                gs=math.ceil(float(float(b1)/bs))
                func(np.int32(b1),bgrad[l],B_grad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
            points+=1
            if(points%batch_size==0):
                points=0
                t += 1
                #executor = ProcessPoolExecutor(batch_size)
                #r_values = []
                #for i in range(batch_size):
                #    future = executor.submit(concurrent_batch, (batchX[i],batchY[i],W,B,L,K))
                #    r_values.append(future.result())
                #    executor.shutdown(wait=True)
                loss.append(error/batch_size)
                error = 0
                for l in range(L):
                    (f1,f2) = wgrad[l].shape
                    b1 = bgrad[l].shape[0]
                    bs=32
                    gs=math.ceil(float(float(f1*f2)/bs))
                    func = mod.get_function("update_1")
                    func(np.int32(f1*f2),np.float32(gamma),np.float32(eta),uw[l],wgrad[l],W[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    gs=math.ceil(float(float(b1)/bs))
                    func(np.int32(b1),np.float32(gamma),np.float32(eta),ub[l],bgrad[l],B[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
                for l in range(L):
                    (f1,f2) = wgrad[l].shape
                    b1 = bgrad[l].shape[0]
                    bs=32
                    gs=math.ceil(float(float(f1*f2)/bs))
                    func = mod.get_function("reset")
                    func(np.int32(f1*f2),wgrad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    gs=math.ceil(float(float(b1)/bs))
                    func(np.int32(b1),bgrad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)   
        e=e+1
        print('Epoch done')
    return W,B,loss

def momentum():
    epoch=1
    batch_size=64
    t,e= 0,0
    W= initialize_weights(xlen,L-1,K,N)
    B= initialize_bias(xlen,L-1,K,N)
    points= 0
    error= 0.0
    gamma= 0.9
    loss=list()
    wgrad=list()
    bgrad=list()
    for l in range(L):
        wgrad.append(np.zeros(shape=np.shape(W[l])).tolist())
        bgrad.append(np.zeros(shape=np.shape(B[l])).tolist())
    uw=list()
    ub=list()
    for l in range(L):
        uw.append(np.zeros(shape=np.shape(W[l])).tolist())
        ub.append(np.zeros(shape=np.shape(B[l])).tolist())
    while (e<epoch):
        for i in range(train_samples):
            H,A,y_hat=feed_forward(trainX[i],W,B,L)
            ei=y_hat[trainY[i]]
            if ei!=0:
                error += -math.log(ei+1e-7)
            W_grad,B_grad=back_propogation(H,A,y_hat,trainY[i],W,L,K)
            W_grad=W_grad[::-1]
            B_grad=B_grad[::-1]
            for l in range(L):
                wgrad[l]=(np.matrix(wgrad[l])+np.matrix(W_grad[l])).tolist()
                bgrad[l]=(bgrad[l]+np.multiply(1,B_grad[l])).tolist()
            points+=1
            if(points%batch_size==0):
                points=0
                t += 1
                loss.append(error/batch_size)
                error=0.0
                for l in range(L):
                    uw[l]=(np.multiply(gamma,uw[l])+np.multiply(eta,wgrad[l])).tolist()
                    W[l]=np.subtract(W[l],uw[l]).tolist()
                    ub[l]=(np.multiply(gamma,ub[l])+np.multiply(eta,bgrad[l])).tolist()
                    B[l]=np.subtract(B[l],ub[l]).tolist()
                wgrad=list()
                bgrad=list()
                for l in range(L):
                    wgrad.append(np.zeros(shape=np.shape(W[l])).tolist())
                    bgrad.append(np.zeros(shape=np.shape(B[l])).tolist())            
        e=e+1
        print('Epoch done')
    return W,B,loss


def nesterov_gpu():
    epoch=1
    batch_size=64
    e,t=0,0
    W= initialize_weights_gpu(xlen,L-1,K,N)
    B= initialize_bias_gpu(xlen,L-1,K,N)
    loss=list()
    points=0
    error=0.0
    gamma=0.9
    wgrad= initialize_weights_zero_gpu(xlen,L-1,K,N)
    bgrad= initialize_bias_gpu(xlen,L-1,K,N)
    uw= initialize_weights_zero_gpu(xlen,L-1,K,N)
    ub= initialize_bias_gpu(xlen,L-1,K,N)
    Wc= initialize_weights_zero_gpu(xlen,L-1,K,N)
    Bc= initialize_bias_gpu(xlen,L-1,K,N)
    while (e<epoch):
        for i in range(train_samples):
            for l in range(L):
                 (f1,f2) = Wc[l].shape
                 b1 = Bc[l].shape[0]
                 bs=32
                 gs=math.ceil(float(float(f1*f2)/bs))
                 func = mod.get_function("lookahead_c")
                 func(np.int32(f1*f2),np.float32(gamma),W[l],uw[l],Wc[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
                 gs=math.ceil(float(float(b1)/bs))
                 func(np.int32(b1),np.float32(gamma),B[l],ub[l],Bc[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
            #H,A,y_hat=feed_forward_gpu(trainX[i],W,B,L)
            #error+=(-math.log(y_hat[trainY[i]]))
            Hi,Ai,y_hati=feed_forward_gpu(trainX[i],Wc,Bc,L)
            error+=(-math.log(y_hati[trainY[i]]))
            W_grad,B_grad=back_propogation_gpu(Hi,Ai,y_hati,trainY[i],Wc,L,K)
            W_grad=W_grad[::-1]
            B_grad=B_grad[::-1]
            for l in range(L):
               (f1,f2) = W_grad[l].shape
               b1 = B_grad[l].shape[0]
               bs=32
               gs=math.ceil(float(float(f1*f2)/bs))
               func = mod.get_function("add")
               func(np.int32(f1*f2),wgrad[l],W_grad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
               gs=math.ceil(float(float(b1)/bs))
               func(np.int32(b1),bgrad[l],B_grad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
            points+=1
            if(points%batch_size==0):
                points=0
                t += 1
                loss.append(error/batch_size)
                error=0.0
                for l in range(L):
                    (f1,f2) = wgrad[l].shape
                    b1 = bgrad[l].shape[0]
                    bs=32
                    gs=math.ceil(float(float(f1*f2)/bs))
                    func = mod.get_function("update_1")
                    func(np.int32(f1*f2),np.float32(gamma),np.float32(eta),uw[l],wgrad[l],W[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    gs=math.ceil(float(float(b1)/bs))
                    func(np.int32(b1),np.float32(gamma),np.float32(eta),ub[l],bgrad[l],B[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
                for l in range(L):
                    (f1,f2) = wgrad[l].shape
                    b1 = bgrad[l].shape[0]
                    bs=32
                    gs=math.ceil(float(float(f1*f2)/bs))
                    func = mod.get_function("reset")
                    func(np.int32(f1*f2),wgrad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    gs=math.ceil(float(float(b1)/bs))
                    func = mod.get_function("reset")
                    func(np.int32(b1),bgrad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)       
        e=e+1
        print('Epoch done')
    return W,B,loss


def nesterov():
    epoch=1
    batch_size=32
    e,t=0,0
    W= initialize_weights(xlen,L-1,K,N)
    B= initialize_bias(xlen,L-1,K,N)
    loss=list()
    points=0
    error=0.0
    gamma=0.9
    wgrad=list()
    bgrad=list()
    for l in range(L):
        wgrad.append(np.zeros(shape=np.shape(W[l])).tolist())
        bgrad.append(np.zeros(shape=np.shape(B[l])).tolist())
    uw=list()
    ub=list()
    for l in range(L):
        uw.append(np.zeros(shape=np.shape(W[l])).tolist())
        ub.append(np.zeros(shape=np.shape(B[l])).tolist())
    while (e<epoch):
        for i in range(train_samples):
            Wc=list()
            Bc=list()
            for l in range(L):
                Wc.append((W[l]-np.multiply(gamma,uw[l])).tolist())
                Bc.append((B[l]-np.multiply(gamma,ub[l])).tolist())
            #H,A,y_hat=feed_forward(trainX[i],W,B,L)
            #error+=(-math.log(y_hat[trainY[i]]))
            Hi,Ai,y_hati=feed_forward(trainX[i],Wc,Bc,L)
            error+=(-math.log(y_hati[trainY[i]]))
            W_grad,B_grad=back_propogation(Hi,Ai,y_hati,trainY[i],Wc,L,K)
            W_grad=W_grad[::-1]
            B_grad=B_grad[::-1]
            for l in range(L):
                wgrad[l]=(np.matrix(wgrad[l])+np.matrix(W_grad[l])).tolist()
                bgrad[l]=(bgrad[l]+np.multiply(1,B_grad[l])).tolist()
            points+=1
            if(points%batch_size==0):
                points=0
                t += 1
                loss.append(error/batch_size)
                error=0.0
                for l in range(L):
                    uw[l]=(np.multiply(gamma,uw[l])+np.multiply(eta,wgrad[l])).tolist()
                    W[l]=np.subtract(W[l],uw[l]).tolist()
                    ub[l]=(np.multiply(gamma,ub[l])+np.multiply(eta,bgrad[l])).tolist()
                    B[l]=np.subtract(B[l],ub[l]).tolist()
                wgrad=list()
                bgrad=list()
                for l in range(L):
                    wgrad.append(np.zeros(shape=np.shape(W[l])).tolist())
                    bgrad.append(np.zeros(shape=np.shape(B[l])).tolist())         
        e=e+1
        print('Epoch done')
    return W,B,loss

def rmsprop_gpu():
    t=0
    epoch=1
    batch_size=64
    W= initialize_weights_gpu(xlen,L-1,K,N)
    B= initialize_bias_gpu(xlen,L-1,K,N)
    loss=list()
    points=0
    error=0.0
    epsilon=1e-10
    beta=0.98
    wgrad= initialize_weights_zero_gpu(xlen,L-1,K,N)
    bgrad= initialize_bias_gpu(xlen,L-1,K,N)
    v_w= initialize_weights_zero_gpu(xlen,L-1,K,N)
    v_b= initialize_bias_gpu(xlen,L-1,K,N)
    while (t<epoch):
        for i in range(train_samples):
            H,A,y_hat=feed_forward_gpu(trainX[i],W,B,L)
            error+=(-math.log(y_hat[trainY[i]]))
            W_grad,B_grad=back_propogation_gpu(H,A,y_hat,trainY[i],W,L,K)
            W_grad=W_grad[::-1]
            B_grad=B_grad[::-1]
            for l in range(L):
                (f1,f2) = wgrad[l].shape
                b1 = bgrad[l].shape[0]
                bs=32
                func = mod.get_function("add")
                gs=math.ceil(float(float(f1*f2)/bs))
                func(np.int32(f1*f2),wgrad[l],W_grad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
                gs=math.ceil(float(float(b1)/bs))
                func(np.int32(b1),bgrad[l],B_grad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
            points+=1
            if(points%batch_size==0):
                points=0
                loss.append(error/batch_size)
                error=0.0
                for l in range(L):
                    (f1,f2) = wgrad[l].shape
                    b1 = bgrad[l].shape[0]
                    bs=32
                    gs=math.ceil(float(float(f1*f2)/bs))
                    func = mod.get_function('update_2')
                    func(np.int32(f1*f2),np.float32(beta),v_w[l],wgrad[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    func = mod.get_function('update_3')
                    func(np.int32(f1*f2),np.float32(epsilon),np.float32(eta),v_w[l],wgrad[l],W[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    gs=math.ceil(float(float(b1)/bs))
                    func = mod.get_function('update_2')
                    func(np.int32(b1),np.float32(beta),v_b[l],bgrad[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    func = mod.get_function('update_3')
                    func(np.int32(b1),np.float32(epsilon),np.float32(eta),v_b[l],bgrad[l],B[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)
                for l in range(L):
                    (f1,f2) = wgrad[l].shape
                    b1 = bgrad[l].shape[0]
                    bs=32
                    gs=math.ceil(float(float(f1*f2)/bs))
                    func = mod.get_function("reset")
                    func(np.int32(f1*f2),wgrad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    gs=math.ceil(float(float(b1)/bs))
                    func = mod.get_function("reset")
                    func(np.int32(b1),bgrad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)             
        t=t+1
        print('Epoch done')
    return W,B,loss


def rmsprop():
    t=0
    epoch=1
    batch_size=64
    W= initialize_weights(xlen,L-1,K,N)
    B= initialize_bias(xlen,L-1,K,N)
    loss=list()
    points=0
    error=0.0
    epsilon=1e-10
    beta=0.98
    wgrad=list()
    bgrad=list()
    for l in range(L):
        wgrad.append(np.zeros(shape=np.shape(W[l])).tolist())
        bgrad.append(np.zeros(shape=np.shape(B[l])).tolist())
    v_w=list()
    v_b=list()
    for l in range(L):
        v_w.append(np.zeros(shape=np.shape(W[l])).tolist())
        v_b.append(np.zeros(shape=np.shape(B[l])).tolist())
    while (t<epoch):
        for i in range(train_samples):
            H,A,y_hat=feed_forward(trainX[i],W,B,L)
            error+=(-math.log(y_hat[trainY[i]]))
            W_grad,B_grad=back_propogation(H,A,y_hat,trainY[i],W,L,K)
            W_grad=W_grad[::-1]
            B_grad=B_grad[::-1]
            for l in range(L):
                wgrad[l]=(np.matrix(wgrad[l])+np.matrix(W_grad[l])).tolist()
                bgrad[l]=(bgrad[l]+np.multiply(1,B_grad[l])).tolist()
            points+=1
            if(points%batch_size==0):
                points=0
                loss.append(error/batch_size)
                error=0.0
                for l in range(L):
                    v_w[l]=(np.multiply(beta,v_w[l])+np.multiply((1-beta),np.square(wgrad[l]))).tolist()
                    tmp=v_w[l].copy()
                    new_t=np.add(epsilon,tmp).tolist()
                    rl=np.reciprocal([[float(j) for j in i] for i in np.sqrt(new_t)]).tolist()
                    W[l]=(W[l]-np.multiply(wgrad[l],np.multiply(eta,rl))).tolist()
                    v_b[l]=(np.multiply(beta,v_b[l])+np.multiply((1-beta),np.square(bgrad[l]))).tolist()
                    tmp1=v_b[l].copy()
                    new_t1=np.add(epsilon,tmp1).tolist()
                    rl1=np.reciprocal([float(i) for i in np.sqrt(new_t1)]).tolist()
                    B[l]=np.subtract(B[l],np.multiply(bgrad[l],np.multiply(eta,rl1))).tolist()
                wgrad=list()
                bgrad=list()
                for l in range(L):
                    wgrad.append(np.zeros(shape=np.shape(W[l])).tolist())
                    bgrad.append(np.zeros(shape=np.shape(B[l])).tolist())            
        t=t+1
        print('Epoch done')
    return W,B,loss

def adam_gpu():
    epoch=1
    t,e=0,0
    batch_size=64
    W= initialize_weights_gpu(xlen,L-1,K,N)
    B= initialize_bias_gpu(xlen,L-1,K,N)
    loss=list()
    points=0
    error=0.0
    epsilon=1e-10
    beta1=0.9
    beta2=0.999
    wgrad= initialize_weights_zero_gpu(xlen,L-1,K,N)
    bgrad= initialize_bias_gpu(xlen,L-1,K,N)
    v_w= initialize_weights_zero_gpu(xlen,L-1,K,N)
    v_b= initialize_bias_gpu(xlen,L-1,K,N)
    m_w= initialize_weights_zero_gpu(xlen,L-1,K,N)
    m_b= initialize_bias_gpu(xlen,L-1,K,N)
    while (e<epoch):
        for i in range(train_samples):
            H,A,y_hat=feed_forward_gpu(trainX[i],W,B,L)
            error+=(-math.log(y_hat[trainY[i]]))
            W_grad,B_grad=back_propogation_gpu(H,A,y_hat,trainY[i],W,L,K)
            W_grad=W_grad[::-1]
            B_grad=B_grad[::-1]
            for l in range(L):
                (f1,f2) = wgrad[l].shape
                b1 = bgrad[l].shape[0]
                bs=32
                func = mod.get_function("add")
                gs=math.ceil(float(float(f1*f2)/bs))
                func(np.int32(f1*f2),wgrad[l],W_grad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
                gs=math.ceil(float(float(b1)/bs))
                func(np.int32(b1),bgrad[l],B_grad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
            points+=1
            if(points%batch_size==0):
                t+=1
                points=0
                loss.append(error/batch_size)
                error=0.0
                for l in range(L):
                    (f1,f2) = wgrad[l].shape
                    b1 = bgrad[l].shape[0]
                    bs=32
                    gs=math.ceil(float(float(f1*f2)/bs))
                    func = mod.get_function("update_4")
                    func(np.int32(f1*f2),np.float32(beta1),m_w[l],wgrad[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    func = mod.get_function("update_2")
                    func(np.int32(f1*f2),np.float32(beta2),v_w[l],wgrad[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    func = mod.get_function("update_5")
                    func(np.int32(f1*f2),np.float32(epsilon),np.float32(eta),np.float32(beta1),np.float32(beta2),np.int32(t),v_w[l],m_w[l],W[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    gs=math.ceil(float(float(b1)/bs))
                    func = mod.get_function("update_4")
                    func(np.int32(b1),np.float32(beta1),m_b[l],bgrad[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    func = mod.get_function("update_2")
                    func(np.int32(b1),np.float32(beta2),v_b[l],bgrad[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    func = mod.get_function("update_5")
                    func(np.int32(b1),np.float32(epsilon),np.float32(eta),np.float32(beta1),np.float32(beta2),np.int32(t),v_b[l],m_b[l],B[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)        
                for l in range(L):
                    (f1,f2) = wgrad[l].shape
                    b1 = bgrad[l].shape[0]
                    bs=32
                    gs=math.ceil(float(float(f1*f2)/bs))
                    func = mod.get_function("reset")
                    func(np.int32(f1*f2),wgrad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    gs=math.ceil(float(float(b1)/bs))
                    func = mod.get_function("reset")
                    func(np.int32(b1),bgrad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0) 
        e=e+1
        print('Epoch done')
    return W,B,loss


def adam():
    t,e=0,0
    epoch=1
    batch_size=64
    W= initialize_weights(xlen,L-1,K,N)
    B= initialize_bias(xlen,L-1,K,N)
    loss=list()
    points=0
    error=0.0
    epsilon=1e-10
    beta1=0.9
    beta2=0.999
    wgrad=list()
    bgrad=list()
    for l in range(L):
        wgrad.append(np.zeros(shape=np.shape(W[l])).tolist())
        bgrad.append(np.zeros(shape=np.shape(B[l])).tolist())
    v_w=list()
    v_b=list()
    for l in range(L):
        v_w.append(np.zeros(shape=np.shape(W[l])).tolist())
        v_b.append(np.zeros(shape=np.shape(B[l])).tolist())
    m_w=list()
    m_b=list()
    for l in range(L):
        m_w.append(np.zeros(shape=np.shape(W[l])).tolist())
        m_b.append(np.zeros(shape=np.shape(B[l])).tolist())
    while (e<epoch):
        for i in range(train_samples):
            H,A,y_hat=feed_forward(trainX[i],W,B,L)
            error+=(-math.log(y_hat[trainY[i]]))
            W_grad,B_grad=back_propogation(H,A,y_hat,trainY[i],W,L,K)
            W_grad=W_grad[::-1]
            B_grad=B_grad[::-1]
            for l in range(L):
                wgrad[l]=(np.matrix(wgrad[l])+np.matrix(W_grad[l])).tolist()
                bgrad[l]=(bgrad[l]+np.multiply(1,B_grad[l])).tolist()
            points+=1
            if(points%batch_size==0):
                t+=1
                points=0
                loss.append(error/batch_size)
                error=0.0
                for l in range(L):
                    m_w[l]=(np.multiply(beta1,m_w[l])+np.multiply((1-beta1),wgrad[l])).tolist()
                    mwe=(np.divide(m_w[l],(1-(beta1**t)))).tolist()
                    v_w[l]=(np.multiply(beta2,v_w[l])+np.multiply((1-beta2),np.square(wgrad[l]))).tolist()
                    vwe=(np.divide(v_w[l],(1-(beta2**t)))).tolist()
                    tmp=vwe.copy()
                    new_t=np.add(epsilon,tmp).tolist()
                    rl=np.reciprocal([[float(j) for j in i] for i in np.sqrt(new_t)]).tolist()
                    W[l]=(W[l]-np.multiply(mwe,np.multiply(eta,rl))).tolist()
                    m_b[l]=(np.multiply(beta1,m_b[l])+np.multiply((1-beta1),bgrad[l])).tolist()
                    mbe=(np.divide(m_b[l],(1-(beta1**t)))).tolist()
                    v_b[l]=(np.multiply(beta2,v_b[l])+np.multiply((1-beta2),np.square(bgrad[l]))).tolist()
                    vbe=(np.divide(v_b[l],(1-(beta2**t)))).tolist()
                    tmp1=vbe.copy()
                    new_t1=np.add(epsilon,tmp1).tolist()
                    rl1=np.reciprocal([float(i) for i in np.sqrt(new_t1)]).tolist()
                    B[l]=np.subtract(B[l],np.multiply(mbe,np.multiply(eta,rl1))).tolist()              
                wgrad=list()
                bgrad=list()
                for l in range(L):
                    wgrad.append(np.zeros(shape=np.shape(W[l])).tolist())
                    bgrad.append(np.zeros(shape=np.shape(B[l])).tolist()) 
        e=e+1
        print('Epoch done')
    return W,B,loss


def nadam_gpu():
    t,e=0,0
    epoch=1
    batch_size=64
    W= initialize_weights_gpu(xlen,L-1,K,N)
    B= initialize_bias_gpu(xlen,L-1,K,N)
    loss=list()
    points=0
    error=0.0
    epsilon=1e-10
    beta1=0.9
    beta2=0.999
    wgrad= initialize_weights_zero_gpu(xlen,L-1,K,N)
    bgrad= initialize_bias_gpu(xlen,L-1,K,N)
    v_w= initialize_weights_zero_gpu(xlen,L-1,K,N)
    v_b= initialize_bias_gpu(xlen,L-1,K,N)
    m_w= initialize_weights_zero_gpu(xlen,L-1,K,N)
    m_b= initialize_bias_gpu(xlen,L-1,K,N)
    while (e<epoch):
        for i in range(train_samples):
            H,A,y_hat=feed_forward_gpu(trainX[i],W,B,L)
            error+=(-math.log(y_hat[trainY[i]]))
            W_grad,B_grad=back_propogation_gpu(H,A,y_hat,trainY[i],W,L,K)
            W_grad=W_grad[::-1]
            B_grad=B_grad[::-1]
            for l in range(L):
                (f1,f2) = wgrad[l].shape
                b1 = bgrad[l].shape[0]
                bs=32
                func = mod.get_function("add")
                gs=math.ceil(float(float(f1*f2)/bs))
                func(np.int32(f1*f2),wgrad[l],W_grad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
                gs=math.ceil(float(float(b1)/bs))
                func(np.int32(b1),bgrad[l],B_grad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
            points+=1
            if(points%batch_size==0):
                t+=1
                points=0
                loss.append(error/batch_size)
                error=0.0
                for l in range(L):
                    (f1,f2) = wgrad[l].shape
                    b1 = bgrad[l].shape[0]
                    bs=32
                    gs=math.ceil(float(float(f1*f2)/bs))
                    func = mod.get_function("update_4")
                    func(np.int32(f1*f2),np.float32(beta1),m_w[l],wgrad[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    func = mod.get_function("update_2")
                    func(np.int32(f1*f2),np.float32(beta2),v_w[l],wgrad[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    func = mod.get_function("update_6")
                    func(np.int32(f1*f2),np.float32(epsilon),np.float32(eta),np.float32(beta1),np.float32(beta2),np.int32(t),v_w[l],m_w[l],wgrad[l],W[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    gs=math.ceil(float(float(b1)/bs))
                    func = mod.get_function("update_4")
                    func(np.int32(b1),np.float32(beta1),m_b[l],bgrad[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    func = mod.get_function("update_2")
                    func(np.int32(b1),np.float32(beta2),v_b[l],bgrad[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    func = mod.get_function("update_6")
                    func(np.int32(b1),np.float32(epsilon),np.float32(eta),np.float32(beta1),np.float32(beta2),np.int32(t),v_b[l],m_b[l],bgrad[l],B[l],block=(bs, 1, 1), grid=(gs, 1), shared=0)  
                for l in range(L):
                    (f1,f2) = wgrad[l].shape
                    b1 = bgrad[l].shape[0]
                    bs=32
                    gs=math.ceil(float(float(f1*f2)/bs))
                    func = mod.get_function("reset")
                    func(np.int32(f1*f2),wgrad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)
                    gs=math.ceil(float(float(b1)/bs))
                    func = mod.get_function("reset")
                    func(np.int32(b1),bgrad[l], block=(bs, 1, 1), grid=(gs, 1), shared=0)  
        e=e+1
        print('Epoch done')
    return W,B,loss

def nadam():
    t,e=0,0
    epoch=1
    batch_size=64
    W= initialize_weights(xlen,L-1,K,N)
    B= initialize_bias(xlen,L-1,K,N)
    loss=list()
    points=0
    error=0.0
    epsilon=1e-10
    beta1=0.9
    beta2=0.999
    wgrad=list()
    bgrad=list()
    for l in range(L):
        wgrad.append(np.zeros(shape=np.shape(W[l])).tolist())
        bgrad.append(np.zeros(shape=np.shape(B[l])).tolist())
    v_w=list()
    v_b=list()
    for l in range(L):
        v_w.append(np.zeros(shape=np.shape(W[l])).tolist())
        v_b.append(np.zeros(shape=np.shape(B[l])).tolist())
    m_w=list()
    m_b=list()
    for l in range(L):
        m_w.append(np.zeros(shape=np.shape(W[l])).tolist())
        m_b.append(np.zeros(shape=np.shape(B[l])).tolist())
    while (e<epoch):
        for i in range(train_samples):
            H,A,y_hat=feed_forward(trainX[i],W,B,L)
            error+=(-math.log(y_hat[trainY[i]]))
            W_grad,B_grad=back_propogation(H,A,y_hat,trainY[i],W,L,K)
            W_grad=W_grad[::-1]
            B_grad=B_grad[::-1]
            for l in range(L):
                wgrad[l]=(np.matrix(wgrad[l])+np.matrix(W_grad[l])).tolist()
                bgrad[l]=(bgrad[l]+np.multiply(1,B_grad[l])).tolist()
            points+=1
            if(points%batch_size==0):
                t+=1
                points=0
                loss.append(error/batch_size)
                error=0.0
                for l in range(L):
                    m_w[l]=(np.multiply(beta1,m_w[l])+np.multiply((1-beta1),wgrad[l])).tolist()
                    mwe=(np.divide(m_w[l],(1-(beta1**t)))).tolist()
                    v_w[l]=(np.multiply(beta2,v_w[l])+np.multiply((1-beta2),np.square(wgrad[l]))).tolist()
                    vwe=(np.divide(v_w[l],(1-(beta2**t)))).tolist()
                    tmp=vwe.copy()
                    new_t=np.add(epsilon,tmp).tolist()
                    rl=np.reciprocal([[float(j) for j in i] for i in np.sqrt(new_t)]).tolist()
                    mwe_up=(np.multiply(beta1,m_w[l])+np.multiply(((1-beta1)/(1-(beta1**t))),wgrad[l])).tolist()
                    W[l]=(W[l]-np.multiply(mwe_up,np.multiply(eta,rl))).tolist()
                    m_b[l]=(np.multiply(beta1,m_b[l])+np.multiply((1-beta1),bgrad[l])).tolist()
                    mbe=(np.divide(m_b[l],(1-(beta1**t)))).tolist()
                    v_b[l]=(np.multiply(beta2,v_b[l])+np.multiply((1-beta2),np.square(bgrad[l]))).tolist()
                    vbe=(np.divide(v_b[l],(1-(beta2**t)))).tolist()
                    tmp1=vbe.copy()
                    new_t1=np.add(epsilon,tmp1).tolist()
                    rl1=np.reciprocal([float(i) for i in np.sqrt(new_t1)]).tolist()
                    mbe_up=(np.multiply(beta1,m_b[l])+np.multiply(((1-beta1)/(1-(beta1**t))),bgrad[l])).tolist()
                    B[l]=np.subtract(B[l],np.multiply(mbe_up,np.multiply(eta,rl1))).tolist()              
                wgrad=list()
                bgrad=list()
                for l in range(L):
                    wgrad.append(np.zeros(shape=np.shape(W[l])).tolist())
                    bgrad.append(np.zeros(shape=np.shape(B[l])).tolist()) 
        e=e+1
        print('Epoch done')
    return W,B,loss


tt_c=list()
tt_g=list()

#No of classes
K=10
#Learning Rate
eta=0.001

#No of Layers
L=3
#size of each hidden layer
N=32

#----------------------------------------
print("Train Algorithm - Stochastic GD")

print("\nGPU Excecution")

begin = time.time()
W_r,B_r,run_tg=stochastic_gd_gpu()
end = time.time()
tt_g.append(end-begin)

#Prediction
valid=0
predict=[]
for i in range(test_samples):
   H,A,y_hat = feed_forward_gpu(testX[i],W_r,B_r,L)
   class_predict = y_hat.index(max(y_hat))
   predict.append(class_predict)
   if(class_predict==testY[i]):
     valid+=1

print ('Accuracy GPU :',(valid/test_samples)*100)
print ('GPU Time :', tt_g[-1])

print("\nCPU Excecution")

begin = time.time()
W_r,B_r,run_tc=stochastic_gd()
end = time.time()
tt_c.append(end-begin)

#Prediction
valid_c=0
predict=[]
for i in range(test_samples):
    H,A,y_hat = feed_forward(testX[i],W_r,B_r,L)
    class_predict = y_hat.index(max(y_hat))
    predict.append(class_predict)
    if(class_predict==testY[i]):
      valid_c+=1

print ('Accuracy CPU :',(valid_c/test_samples)*100)
print ('CPU Time :', tt_c[-1])

axes = plt.gca()
axes.yaxis.grid()
plt.plot(np.arange(1,len(run_tg)+1),run_tg)
plt.plot(np.arange(1,len(run_tc)+1),run_tc)
plt.title("Train Loss Stochastic GD")
plt.show()

print("--------------------------------------------")

print("Train Algorithm - Momentum GD")

print("\nGPU Excecution")

begin = time.time()
W_r,B_r,run_tg=momentum_gpu()
end = time.time()
tt_g.append(end-begin)

#Prediction
valid=0
predict=[]
for i in range(test_samples):
   H,A,y_hat = feed_forward_gpu(testX[i],W_r,B_r,L)
   class_predict = y_hat.index(max(y_hat))
   predict.append(class_predict)
   if(class_predict==testY[i]):
     valid+=1

print ('Accuracy GPU :',(valid/test_samples)*100)
print ('GPU Time :', tt_g[-1])

print("\nCPU Excecution")

begin = time.time()
W_r,B_r,run_tc=momentum()
end = time.time()
tt_c.append(end-begin)

#Prediction
valid_c=0
predict=[]
for i in range(test_samples):
    H,A,y_hat = feed_forward(testX[i],W_r,B_r,L)
    class_predict = y_hat.index(max(y_hat))
    predict.append(class_predict)
    if(class_predict==testY[i]):
      valid_c+=1

print ('Accuracy CPU :',(valid_c/test_samples)*100)
print ('CPU Time :', tt_c[-1])

axes = plt.gca()
axes.yaxis.grid()
plt.plot(np.arange(1,len(run_tg)+1),run_tg)
plt.plot(np.arange(1,len(run_tc)+1),run_tc)
plt.title("Train Loss Momentum GD")
plt.show()

print("--------------------------------------------")
print("Train Algorithm - Nesterov GD")

print("\nGPU Excecution")

begin = time.time()
W_r,B_r,run_tg=nesterov_gpu()
end = time.time()
tt_g.append(end-begin)

#Prediction
valid=0
predict=[]
for i in range(test_samples):
   H,A,y_hat = feed_forward_gpu(testX[i],W_r,B_r,L)
   class_predict = y_hat.index(max(y_hat))
   predict.append(class_predict)
   if(class_predict==testY[i]):
     valid+=1

print ('Accuracy GPU :',(valid/test_samples)*100)
print ('GPU Time :', tt_g[-1])

print("\nCPU Excecution")

begin = time.time()
W_r,B_r,run_tc=nesterov()
end = time.time()
tt_c.append(end-begin)

#Prediction
valid_c=0
predict=[]
for i in range(test_samples):
    H,A,y_hat = feed_forward(testX[i],W_r,B_r,L)
    class_predict = y_hat.index(max(y_hat))
    predict.append(class_predict)
    if(class_predict==testY[i]):
      valid_c+=1

print ('Accuracy CPU :',(valid_c/test_samples)*100)
print ('CPU Time :', tt_c[-1])

axes = plt.gca()
axes.yaxis.grid()
plt.plot(np.arange(1,len(run_tg)+1),run_tg)
plt.plot(np.arange(1,len(run_tc)+1),run_tc)
plt.title("Train Loss Nesterov GD")
plt.show()

print("--------------------------------------------")
print("Train Algorithm - RmsProp GD")

print("\nGPU Excecution")

begin = time.time()
W_r,B_r,run_tg=rmsprop_gpu()
end = time.time()
tt_g.append(end-begin)

#Prediction
valid=0
predict=[]
for i in range(test_samples):
   H,A,y_hat = feed_forward_gpu(testX[i],W_r,B_r,L)
   class_predict = y_hat.index(max(y_hat))
   predict.append(class_predict)
   if(class_predict==testY[i]):
     valid+=1

print ('Accuracy GPU :',(valid/test_samples)*100)
print ('GPU Time :', tt_g[-1])

print("\nCPU Excecution")

begin = time.time()
W_r,B_r,run_tc=rmsprop()
end = time.time()
tt_c.append(end-begin)

#Prediction
valid_c=0
predict=[]
for i in range(test_samples):
    H,A,y_hat = feed_forward(testX[i],W_r,B_r,L)
    class_predict = y_hat.index(max(y_hat))
    predict.append(class_predict)
    if(class_predict==testY[i]):
      valid_c+=1

print ('Accuracy CPU :',(valid_c/test_samples)*100)
print ('CPU Time :', tt_c[-1])

axes = plt.gca()
axes.yaxis.grid()
plt.plot(np.arange(1,len(run_tg)+1),run_tg)
plt.plot(np.arange(1,len(run_tc)+1),run_tc)
plt.title("Train Loss RMSProp")
plt.show()

print("--------------------------------------------")
print("Train Algorithm - Adam GD")

print("\nGPU Excecution")

begin = time.time()
W_r,B_r,run_tg=adam_gpu()
end = time.time()
tt_g.append(end-begin)

#Prediction
valid=0
predict=[]
for i in range(test_samples):
   H,A,y_hat = feed_forward_gpu(testX[i],W_r,B_r,L)
   class_predict = y_hat.index(max(y_hat))
   predict.append(class_predict)
   if(class_predict==testY[i]):
     valid+=1

print ('Accuracy GPU :',(valid/test_samples)*100)
print ('GPU Time :', tt_g[-1])

print("\nCPU Excecution")

begin = time.time()
W_r,B_r,run_tc=adam()
end = time.time()
tt_c.append(end-begin)

#Prediction
valid_c=0
predict=[]
for i in range(test_samples):
    H,A,y_hat = feed_forward(testX[i],W_r,B_r,L)
    class_predict = y_hat.index(max(y_hat))
    predict.append(class_predict)
    if(class_predict==testY[i]):
      valid_c+=1

print ('Accuracy CPU :',(valid_c/test_samples)*100)
print ('CPU Time :', tt_c[-1])

axes = plt.gca()
axes.yaxis.grid()
plt.plot(np.arange(1,len(run_tg)+1),run_tg)
plt.plot(np.arange(1,len(run_tc)+1),run_tc)
plt.title("Train Loss Adam")
plt.show()

print("--------------------------------------------")
print("Train Algorithm - Nadam GD")

print("\nGPU Excecution")

begin = time.time()
W_r,B_r,run_tg=nadam_gpu()
end = time.time()
tt_g.append(end-begin)

#Prediction
valid=0
predict=[]
for i in range(test_samples):
   H,A,y_hat = feed_forward_gpu(testX[i],W_r,B_r,L)
   class_predict = y_hat.index(max(y_hat))
   predict.append(class_predict)
   if(class_predict==testY[i]):
     valid+=1

print ('Accuracy GPU :',(valid/test_samples)*100)
print ('GPU Time :', tt_g[-1])

print("\nCPU Excecution")

begin = time.time()
W_r,B_r,run_tc=nadam()
end = time.time()
tt_c.append(end-begin)

#Prediction
valid_c=0
predict=[]
for i in range(test_samples):
    H,A,y_hat = feed_forward(testX[i],W_r,B_r,L)
    class_predict = y_hat.index(max(y_hat))
    predict.append(class_predict)
    if(class_predict==testY[i]):
      valid_c+=1

print ('Accuracy CPU :',(valid_c/test_samples)*100)
print ('CPU Time :', tt_c[-1])

axes = plt.gca()
axes.yaxis.grid()
plt.plot(np.arange(1,len(run_tg)+1),run_tg)
plt.plot(np.arange(1,len(run_tc)+1),run_tc)
plt.title("Train Loss Nadam")
plt.show()

print("--------------------------------------------")
label = ['Stochastic','Momentum','Nesterov','RmsProp','Adam','Nadam']
axes = plt.gca()
axes.yaxis.grid()
plt.xticks(range(len(label)), label, size='small')
plt.plot(tt_c,label='CPU')
plt.plot(tt_g,label='GPU')
plt.title("Train Time - GPU vs CPU")
plt.xlabel('Optimization Algorithm')
plt.ylabel('Train Time')
plt.show()
#-------------------------------------------
