import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

Ned=3
axes = np.loadtxt('%dd/axes.txt'%Ned,skiprows=1)
logEs = axes[:,0]
logTs = axes[:,1]
specData = np.loadtxt('%dd/gamma.txt'%Ned,skiprows=0).T

#spectrumInterp = interpolate.interp2d(logEs , logTs , specData)
#print(spectrumInterp(-4.9, -1))
spectrumInterp = interpolate.interp2d(logTs , logEs , specData)
print(spectrumInterp(-1, -4.9))