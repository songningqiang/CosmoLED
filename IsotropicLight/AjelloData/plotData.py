import numpy as np
from matplotlib import pyplot as plt

#egretData = np.loadtxt('../GammaData/EGRET-2004.csv',delimiter=',',skiprows=1)

##EmidEgret = egretData[:,0]
#EwidthEgret = (egretData[:,4]-egretData[:,3])
#EmidEgret = (egretData[:,4]+egretData[:,3])/2
#IEgret = egretData[:,1] #* EwidthEgret
#dIEgret = egretData[:,2]#* EwidthEgret
#print(EmidEgret)
#plt.errorbar(EmidEgret, IEgret, dIEgret, EwidthEgret/2, fmt='none')

#egretData = np.loadtxt('EGRET.dat',delimiter=' ',skiprows=3)

#EwidthEgret = 1e-3*(egretData[:,2]+egretData[:,1])
#EmidEgret = 1e-3*egretData[:,0]
#IEgret = 1e-3*egretData[:,3] / EmidEgret**2
#dIEgret = 1e-3*egretData[:,4] / EmidEgret**2
#plt.errorbar(EmidEgret, IEgret, dIEgret, EwidthEgret/2, fmt='none')


swiftData = np.loadtxt('../GammaData/SWIFT_BAT-2008.csv',delimiter=',',skiprows=1)

#EmidSwift = swiftData[:,0]
EwidthSwift = (swiftData[:,4]-swiftData[:,3])
EmidSwift = (swiftData[:,4]+swiftData[:,3])/2
ISwift = swiftData[:,1] #* EwidthSwift
dISwift = swiftData[:,2]#* EwidthSwift
plt.errorbar(EmidSwift, ISwift, dISwift, EwidthSwift/2, fmt='none')

swiftData = np.loadtxt('BAT.dat',delimiter=' ',skiprows=3)

EwidthSwift = 1e-3*(swiftData[:,2]+swiftData[:,1])
EmidSwift = 1e-3*swiftData[:,0]
ISwift = 1e-3*swiftData[:,3] / EmidSwift**2
dISwift = 1e-3*swiftData[:,4] / EmidSwift**2
plt.errorbar(EmidSwift, ISwift, dISwift, EwidthSwift/2, fmt='none')

#nagoyaData = np.loadtxt('Nagoya-1975.csv',delimiter=',',skiprows=1)

##EmidNagoya = nagoyaData[:,0]
#EwidthNagoya = (nagoyaData[:,4]-nagoyaData[:,3])
#EmidNagoya = (nagoyaData[:,4]+nagoyaData[:,3])/2
#INagoya = nagoyaData[:,1] #* EwidthNagoya
#dINagoya = nagoyaData[:,2]#* EwidthNagoya

#comptelData = np.loadtxt('COMPTEL-2000.csv',delimiter=',',skiprows=1)

##EmidComptel = comptelData[:,0]
#EwidthComptel = (comptelData[:,4]-comptelData[:,3])
#EmidComptel = (comptelData[:,4]+comptelData[:,3])/2
#IComptel = comptelData[:,1] #*EwidthComptel
#dIComptel = comptelData[:,2]#*EwidthComptel

#fermiData = np.genfromtxt('FERMI_LAT-2015.txt',skip_header=61,skip_footer=52)

#EmidFermiA = (fermiData[:,2] + fermiData[:,1])/2
#EwidthFermiA = (fermiData[:,2] - fermiData[:,1])
#IFermiA = fermiData[:,3]/EwidthFermiA
#dIFermiA = np.array([np.sqrt(fermiData[:,4]**2 + fermiData[:,6]**2), np.sqrt(fermiData[:,5]**2 + fermiData[:,7]**2)])/EwidthFermiA 

##fermiData = np.genfromtxt('FERMI_LAT-2015.txt',skip_header=87,skip_footer=26)

##EmidFermiB = (fermiData[:,2] + fermiData[:,1])/2
##EwidthFermiB = (fermiData[:,2] - fermiData[:,1])
##IFermiB = fermiData[:,3]
##dIFermiB = np.array([fermiData[:,4] + fermiData[:,6], fermiData[:,5] + fermiData[:,7]])

##fermiData = np.genfromtxt('FERMI_LAT-2015.txt',skip_header=113,skip_footer=0)

##EmidFermiC = (fermiData[:,2] + fermiData[:,1])/2
##EwidthFermiC = (fermiData[:,2] - fermiData[:,1])
##IFermiC = fermiData[:,3]
##dIFermiC = np.array([fermiData[:,4] + fermiData[:,6], fermiData[:,5] + fermiData[:,7]])

#plt.errorbar(EmidChandra, IChandra, dIChandra, EwidthChandra/2, fmt='none')
#plt.errorbar(EmidSwift, ISwift, dISwift, EwidthSwift/2, fmt='none')
#plt.errorbar(EmidNagoya, INagoya, dINagoya, EwidthNagoya/2, fmt='none')
#plt.errorbar(EmidComptel, IComptel, dIComptel, EwidthComptel/2, fmt='none')
#plt.errorbar(EmidEgret, IEgret, dIEgret, EwidthEgret/2, fmt='none')
#plt.errorbar(EmidFermiA, IFermiA, dIFermiA, EwidthFermiA/2, fmt='none')#, capsize=2,linewidth=1)
#plt.errorbar(EmidFermiB, IFermiB, dIFermiB, EwidthFermiB/2, fmt='.', capsize=2,linewidth=1)
#plt.errorbar(EmidFermiC, IFermiC, dIFermiC, EwidthFermiC/2, fmt='.', capsize=2,linewidth=1)
plt.legend(['Chandra/XMM-Newton','Swift-BAT','Nagoya','COMPTEL', 'EGRET','FERMI-LAT'])
plt.xlabel('E (MeV)',size=14)
plt.ylabel('J (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)',size=14)
plt.yscale('log')
plt.xscale('log')
plt.show()
