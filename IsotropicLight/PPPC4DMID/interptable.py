import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

d3 = np.loadtxt('3d/gamma.txt')
daxes = np.loadtxt('3d/axes.txt')
lgE = daxes[:,0]
E = 10.**lgE
lgTBH = daxes[:,1]
#TT, EE = np.meshgrid(lgTBH, lgE)
EE, TT = np.meshgrid(lgE, lgTBH)
#f = interpolate.interp2d(TT, EE, d3, bounds_error = True)
#dNdE = d3/np.log(10)/E[:,None]
dNdE = d3
#f = interpolate.interp2d(TT, EE, dNdE, bounds_error = True)
f = interpolate.interp2d(EE, TT, dNdE, bounds_error = True)

Ttest = 0.1
lgTtest = np.log10(Ttest)
lgEtest = lgE
Etest = 10.**lgEtest
#dNdlgE = f(lgTtest, -4.9)
dNdlgE = f(-4.9, lgTtest)
print dNdlgE
exit(1)
dNdE = dNdlgE/np.log(10.)/Etest
plt.loglog(Etest, dNdE)
plt.show()


