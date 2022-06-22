import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate
from spectrum_table import PBH_dMdt
from spectrum_table import PBH_primary_spectrum
from spectrum_table import get_temperature_from_mass

pi = np.pi
ztSetup = False #indicates whether data for z(t) interpolation has been setup

#Planck 2018
Omm = 0.3153
OmL = 0.6847
Omgamma = 5.38e-5
Omr = 1.68 * Omgamma #Include effect of neutrinos
h = 0.6736

#PDG
Omdm = 0.265
rhoc0_gcm3 = 1.87834e-29 * h**2 #critical density today in g/cm^3
me = 511e3 #electron mass eV
mmu = 106e6 #muon mass eV
mpi0 = 135e6 #neutral pion mass eV
mpipm = 140e6 #charged pion mass eV

LamQCD = 300e6 #QCD cutoff

#Physical Constants and Unit Conversions
sigmaT_eV = 1.707e-15 #Thompson cross-section in eV^-2
G_eV = 6.708e-57 #Gravitational constnat in eV^-2
Mstar0 = 1.16e66 #eV/solarMass
eVperg = 5.6096e32 #eV/g
c = 3e8 #m/s
hbar = 6.582e-16 #eV * s
kmperm = 1e-3 #km/m
mpercm = 1e-2 #m/cm
kmperMpc = 3.086e19 #km per Megaparsec
eVMpc =  kmperm * c * hbar/kmperMpc  #eV*Mpc (converts natural units to Mpc)
GeVpereV = 1e-9 #GeV/eV
MpcperGpc = 1e3
speryear = 3.154e7 #seconds/year

#Change in black hole mass in eV/s
#Mass M in eV, has Ned large extra dimensions with scale of gravity Mstar (eV)
def dMdtED(t, M, Ned, Mstar):
	M_g = M / eVperg

	Mstar_GeV = Mstar * GeVpereV
	
	#Integrator uses M as a vector but PBH_dMdt is not vectorized
	dmdt_gs = np.zeros(M_g.size)
	for i, Mi in enumerate(M_g):
		dmdt_gs[i] = PBH_dMdt(Mi, t, Ned, Mstar_GeV) #Ningqiang function in g/s

	return dmdt_gs * eVperg

#Calculate black hole mass for a function of time (seconds) for a given initial mass (eV)
def calcMoft(tIn, M0, Ned = 0, Mstar = 0, odeMethod = 'BDF'):
	ti = 1 #integration start time (very early universe)
	tf = 15e9 * speryear #integration end time (after today)

	if Ned > 0:
		sol = integrate.solve_ivp(dMdtED,t_span=[ti,tf],y0=[M0],method=odeMethod, args=(Ned, Mstar))
	else:
		sol = integrate.solve_ivp(dMdt4D,t_span=[ti,tf],y0=[M0],method=odeMethod)
	tvals = sol['t']	
	Ms = sol['y'][0]

	Msout = np.interp(tIn, tvals, Ms,right=0) #interpolate to input z
