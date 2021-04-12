#!/usr/bin/env python3
"""
A script to run nflow in HPC, like eofe cluster
"""
import pickle5 as pickle

#Standard import statements
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
from datetime import datetime

#Pytorch imports
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch import distributions
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
from torch.nn.parameter import Parameter

#NFlow library imports
from nflib.flows import (
    AffineConstantFlow, AffineHalfFlow, MLP, 
    NormalizingFlow, NormalizingFlowModel,
)

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2

#Create data class
class dataXZ:
  """
  read the data stored in pickle format
  the converting routine is at https://github.com/6862-2021SP-team3/hipo2pickle
  """
  def __init__(self):
    with open('pi0.pkl', 'rb') as f:
        xz = np.array(pickle.load(f), dtype=np.float32)
        xz = xz[:, 1:]
        x = xz[:, :16]
        z = xz[:, 16:]
        xwithoutPid = x[:, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]]
        zwithoutPid = z[:, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]]
        self.xz = xz
        self.x = torch.from_numpy(np.array(x))
        self.z = torch.from_numpy(np.array(z))
        self.xwithoutPid = torch.from_numpy(xwithoutPid)
        self.zwithoutPid = torch.from_numpy(zwithoutPid)

  def sample(self, n):
        randint = np.random.randint( self.xz.shape[0], size =n)
        xz = self.xz[randint]
        x = self.x[randint]
        z = self.z[randint]
        xwithoutPid = self.xwithoutPid[randint]
        zwithoutPid = self.zwithoutPid[randint]
        return {"xz":xz, "x": x, "z": z, "xwithoutPid": xwithoutPid, "zwithoutPid": zwithoutPid}

xz = dataXZ()

# construct a model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# try with electron momentum magintude and polar angle only
prior = TransformedDistribution(Uniform(torch.zeros(2, device = device), torch.ones(2, device = device)), SigmoidTransform().inv) # Logistic distribution
#prior = MultivariateNormal(torch.zeros(2, device = device), torch.eye(2, device = device)) # Normal distribution
# NICE
flows = [AffineHalfFlow(dim=2, parity=i%2, scale=False) for i in range(12)]
#print(flows)
flows.append(AffineConstantFlow(dim=2, shift=False))
#print(flows)


# construct the model

model = NormalizingFlowModel(prior, flows, device = device)
model.to(device)

# optimizer
#optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-9) # this one was pretty good, but oscillates
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-9) # pretty solid, two bands
#optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-10) # weird tail at high electron momenutm
#optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-9) # 


print("number of params: ", sum(p.numel() for p in model.parameters()))

# in training mode to learn the distribution.
model.train()
start_now = datetime.now()
start_time = start_now.strftime("%H:%M:%S")
print("Start Time =", start_time)
losses = []
for k in range(5000):
    sampleDict = xz.sample(1000)
    x = sampleDict["xwithoutPid"][:, 0:2] # try with electron momentum magintude and polar angle only.
    x = x.to(device)
    zs, prior_logprob, log_det = model(x)
    logprob = prior_logprob + log_det
    loss = -torch.sum(logprob) # NLL

    model.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if k % 100 == 0:
        run_time = datetime.now()
        elapsedTime = (run_time - start_now )
        print("On step {} - loss {:.2f}, Current Running Time = {:.2f} seconds".format(k,loss.item(),elapsedTime.total_seconds())) 

now = datetime.now()
end_time = now.strftime("%H:%M:%S")
print("End Time =", end_time)
elapsedTime = (now - start_now )
print("Total Run Time = {:.5f} seconds".format(elapsedTime.total_seconds()))


fig, ax = plt.subplots(figsize =(10, 7)) 
#print(np.arange(len(losses)))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "16"

plt.scatter(np.arange(len(losses)),losses, c='g', s=20)
plt.title('Loss vs. Training Step')
ax.set_xlabel("Training Step")  
ax.set_ylabel("Loss")
plt.tight_layout()
plt.savefig("loss.pdf")

# start testing
model.eval()
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "16"

sampleDict = xz.sample(10000)
x = sampleDict["xwithoutPid"][:, 0:2]
x = x.to(device)
zs, prior_logprob, log_det = model(x)
z = zs[-1]

p = model.prior.sample([10000, 2]).squeeze()
if device == "cpu":
  x = x.detach().numpy()
  z = z.detach().numpy()
else:
  x = x.cpu().detach().numpy()
  z = z.cpu().detach().numpy()
  p = p.cpu()

fig, ax = plt.subplots(figsize =(10, 7)) 
plt.scatter(p[:,0], p[:,1], c='g', s=5)
plt.scatter(z[:,0], z[:,1], c='r', s=5)
plt.scatter(x[:,0], x[:,1], c='b', s=5)
plt.legend(['prior', 'x->z', 'data'])
plt.axis('scaled')
plt.title('x -> z')


zs = model.sample(10000)
z = zs[-1]
if device == "cpu":
  z = z.detach().numpy()
else:
  z = z.cpu().detach().numpy()
fig, ax = plt.subplots(figsize =(10, 7)) 
plt.scatter(z[:,0], z[:,1], c='r', s=5, alpha=0.5)
plt.scatter(x[:,0], x[:,1], c='b', s=5, alpha=0.5)
plt.legend(['NFlow Model','Physics Model'])
plt.title('NFlow Generated Data vs. Physics Model Training Data')
ax.set_xlabel("Electron Momentum")  
ax.set_ylabel("Polar Angle")

fig, ax = plt.subplots(figsize =(10, 7)) 
ax.set_xlabel("Electron Momentum")  
ax.set_ylabel("Polar Angle")
#plt.scatter(x[:,0], x[:,1], c='g', s=5)
plt.title('Electron Momentum vs. Angle, Physics Model')
plt.hist2d(x[:,0], x[:,1],bins =[40, 40],norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
plt.colorbar()


fig, ax = plt.subplots(figsize =(10, 7)) 
ax.set_xlabel("Electron Momentum")  
ax.set_ylabel("Polar Angle")
plt.title('Electron Momentum vs. Angle, NFlow Generated')
#plt.scatter(x[:,0], x[:,1], c='g', s=5)
plt.hist2d(z[:,0], z[:,1],bins =[40, 40],norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
plt.colorbar()
plt.savefig("output.pdf")