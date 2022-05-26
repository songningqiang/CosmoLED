import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import interp2d
from scipy.interpolate import interpn
import math

from ctypes import *
import time



pi = np.pi


#PDG
Omdm = 0.265

me = 511e3 #electron mass eV
mmu = 106e6 #muon mass eV
mpi0 = 135e6 #neutral pion mass eV
mpipm = 140e6 #charged pion mass eV
Mpl = 1.22089e28  #4D Planck mass in eV
T0_Kelvin = 2.7255 #Temperature today in kelvin

LamQCD = 300e6 #QCD cutoff

#arxiv:1912.04296
rhoNFW = 2.2e-24 / 4#Milky Way NFW fit density in g/cm^3
rNFW = 9.98 #Milky Way NFW fit radius kpc

#arxiv:1904.05721
rEarth = 8.178 #kpc 

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
kpcPerMpc = 1e3 #kpc/Mpc
eVMpc =  kmperm * c * hbar/kmperMpc  #eV*Mpc (converts natural units to Mpc)
GeVpereV = 1e-9 #GeV/eV
GeVperkeV = 1e-6 #GeV/eV	
GeVperMeV = 1e-3 #GeV/MeV
MpcperGpc = 1e3
speryear = 3.154e7 #seconds/year
kB = 8.617330e-5 #eV/K


#define constants
gramtoGeV = 5.62e23
hbar = 6.582e-25 #GeV*s

def kn(n):
	return (2.**n*math.pi**((n-3.)/2.)*math.gamma((n+3.)/2.)/(n+2.))**(1./(n+1.))

def get_radius_from_mass(mass,n,Mstar):
	#mass is in gram
	M = mass*gramtoGeV
	return kn(n)/Mstar*(M/Mstar)**(1./(n+1.))

def get_temperature_from_mass(mass,n,Mstar):
	return (n+1.)/(4.*math.pi*get_radius_from_mass(mass,n,Mstar))

def get_mass_from_temperature(T_BH,n,Mstar):
	return Mstar*((n+1.)/(4.*math.pi*kn(n))*Mstar/T_BH)**(n+1)/gramtoGeV	

#print(get_temperature_from_mass(1.e13,4,1e4))
#exit()

#particle_dict has structure 'particle_name':[mass,spin,dof,sigmoid_factor]
#neutrino is the sum of all flavors, particle/antiparticle are separated
particle_dict = {'gamma': [0,1.0,2,0],
			  'neutrino': [0,0.5,3,0],
			  'electron': [5.110e-4,0.5,2,0],
				  'muon': [1.057e-1,0.5,2,0],
				   'tau': [1.777,0.5,2,0],
					'up': [2.2e-3,0.5,6,1],
				  'down': [4.7e-3,0.5,6,1],
				 'charm': [1.28,0.5,6,1],
			   'strange': [9.6e-2,0.5,6,1],
				   'top': [173.1,0.5,6,1],
				'bottom': [4.18,0.5,6,1],
			   'w-boson': [80.39,1.0,3,0],
			   'z-boson': [91.19,1.0,3,0],
				 'gluon': [6e-1,1.0,16,1],
				 'higgs': [125.09,0.0,1,0],
				   'pi0': [1.350e-1,0.0,1,-1],
				  'piCh': [1.396e-1,0.0,1,-1]
}



def QCD_factor(T_BH, sigmoid_factor):
	Lam_QCD = 0.3
	sigma = 0.1
	return 1. / (1. + np.exp(- sigmoid_factor*(np.log10(T_BH) - np.log10(Lam_QCD))/sigma))



#find integrated emission power
xi_scalar = np.array([0.0167,0.0675,0.1868,0.4156,0.8021,1.4011])
xi_fermion = np.array([0.0146,0.0612,0.1666,0.3619,0.6836,1.1736])
xi_vector = np.array([0.0115,0.0611,0.1864,0.4316,0.8469,1.4884])
xi_graviton = np.array([0.0097235,0.099484,0.4927,1.9042,6.8861,24.6844])
b_scalar = np.array([0.3334,0.2832,0.2806,0.2846,0.2932,0.3044])
b_fermion = np.array([0.2758,0.2933,0.2879,0.2895,0.2959,0.2739])
b_vector = np.array([0.2203,0.2638,0.2743,0.2583,0.2975,0.3105])
c_scalar = np.array([1.236,1.291,1.296,1.292,1.282,1.27])
c_fermion = np.array([1.297,1.279,1.286,1.284,1.276,1.303])
c_vector = np.array([1.361,1.311,1.303,1.329,1.279,1.265])

def PBH_dMdt(PBH_mass, time, n, Mstar):
	u"""Returns the differential mass loss rate of an primordial black hole with
	a mass of :code:`PBH_mass*scale` gramm.

	This method calculates the differential mass loss rate at a given (rescaled) mass
	of the black hole and at a given time, given by

	.. math::

	   \\frac{\\mathrm{d}M\,\\mathrm{[g]}}{\\mathrm{d}t} = - 5.34\\cdot 10^{25}
	   \\cdot \\mathcal{F}(M) \\cdot \\left(\\frac{1}{M\,\\mathrm{[g]}}\\right)^2
	   \,\\frac{\\mathrm{1}}{\\mathrm{s}}

	.. note::

	   Even if this method is not using the variable :code:`time`, it is needed for the
	   ODE-solver within the :meth:`PBH_mass_at_t <DarkAges.evaporator.PBH_mass_at_t>`
	   for the correct interpretation of the differential equation for the mass of
	   the black hole.

	Parameters
	----------
	PBH_mass : :obj:`float`
		(Rescaled) mass of the black hole in units of :math:`\\mathrm{scale} \\cdot \\mathrm{g}`
	time : :obj:`float`
		Time in units of seconds. (Not used, but needed for the use with an ODE-solver)
	scale : :obj:`float`, *optional*
		For a better numerical performance the differential equation can be expressed in
		terms of a different scale than :math:`\\mathrm{g}`. For example the choice
		:code:`scale = 1e10` returns the differential mass loss rate in units of
		:math:`10^{10}\\mathrm{g}`. This parameter is optional. If not given, :code:`scale = 1`
		is assumed.

	Returns
	-------
	:obj:`float`
		Differential mass loss rate in units of :math:`\\frac{\\mathrm{g}}{\\mathrm{s}}`
	"""

	if PBH_mass > 0:
	#if PBH_mass > Mstar/gramtoGeV:
		ret = 0.
		rh = get_radius_from_mass(PBH_mass, n, Mstar)
		TH = (n+1.)/(4.*np.pi*rh)
		for ptype in particle_dict.keys():
			pmass = particle_dict.get(ptype)[0]
			spin = particle_dict.get(ptype)[1]
			dof = particle_dict.get(ptype)[2]
			sigmoid_factor = particle_dict.get(ptype)[3]
			#add particle and antiparticle
			if spin == 0.5 or ptype == 'w-boson' or ptype == 'piCh':
				dof *= 2.

			if spin == 0.0:
				xi = xi_scalar[n-1]
				b = b_scalar[n-1]
				c = c_scalar[n-1]
			elif spin == 0.5:
				xi = xi_fermion[n-1]
				b = b_fermion[n-1]
				c = c_fermion[n-1]
			elif spin == 1.0:
				xi = xi_vector[n-1]
				b = b_vector[n-1]
				c = c_vector[n-1]
			else:
				raise TypeError('The spin "{:1.1}" is not recognized'.format(spin))

			if sigmoid_factor != 0:
				ret += dof*xi*np.exp(-b*(pmass/TH)**c)*QCD_factor(TH,sigmoid_factor)
			else:
				ret += dof*xi*np.exp(-b*(pmass/TH)**c)	

		#add gravitons
		ret += xi_graviton[n-1]
		ret += ret/(2.*np.pi)/rh**2

		return -ret/gramtoGeV/hbar
	else:
		return -0.0	


def dMdtED(t, M, Ned, Mstar):
	"""\\frac{dM}{dt}, change in black hole mass in eV/s in case of large extra dimensions

	Parameters
	----------
	t    : float
		Time (seconds) used as independent variable for integrator
	M    : float
		Mass of black hole in eV
	Ned  : int
		Number of large extra dimensions
	Mstar: float
		Scale of gravity (eV)

	Returns
	-------
	float
		Instatneous mass loss in eV/s
	"""

	M_g = M / eVperg

	Mstar_GeV = Mstar * GeVpereV
	
	#Integrator uses M as a vector but PBH_dMdt is not vectorized
	dmdt_gs = np.zeros(M_g.size)
	for i, Mi in enumerate(M_g):
		dmdt_gs[i] = PBH_dMdt(Mi, t, Ned, Mstar_GeV) #Ningqiang's function in g/s

	return dmdt_gs * eVperg


def calcMoft(tIn, M0, Ned = 0, Mstar = 0, odeMethod = 'BDF'):
	"""Calculate black hole mass for a function of time (seconds) for a given initial mass (eV)

	Parameters
	----------
	tIn  : numpy array
		Ages of the universe (seconds) for the black holes  masses to be evaluated at.
		Cannot be earlier than 1 second or later than 15 Billion years
	M0   : float
		Initial mass of black hole in eV
	Ned  : int
		Number of large extra dimensions. The default value is 0 for 4D case  
	Mstar: float
		Scale of gravity (eV). Irrelevant for 4D case but must be set in case
		with Ned > 0
	odeMethod : string
		Method of integration. See scipy.integrate.solve_ivp for list of options

	Returns
	-------
	numpy array
		List of black hole masses (eV) at times tIn
	"""
	ti = 1 #integration start time (very early universe)
	tf = 15e9 * speryear #integration end time (after today)

	if Ned > 0:
		sol = integrate.solve_ivp(dMdtED,t_span=[ti,tf],y0=[M0],method=odeMethod, args=(Ned, Mstar))
	else:
		sol = integrate.solve_ivp(dMdt4D,t_span=[ti,tf],y0=[M0],method=odeMethod)
	tvals = sol['t']	
	Ms = sol['y'][0]
	tevap = tvals[Ms > 0][-1]
	Mevap = Ms[Ms > 0][-1]

	Msout = np.interp(tIn, tvals, Ms,right=0) #interpolate to input z

	#No negative masses
	Msout = np.maximum.reduce([Msout,np.zeros(Msout.size)])

	return Msout#, tevap, Mevap

def lifetime(Mbh, Ned):
    Nz = 5000
    zmin = 1e-2
    zmax = 1100 #decoupling

    zs = np.logspace(np.log10(zmax), np.log10(zmin), Nz) #redshifts
    ts = tofz(zs) #time in seconds
	
    Ms = calcMoft(ts, Mbh, Ned, 1e12, odeMethod = 'BDF') #Black hole masses in eV
    return ts[Ms > 0][-1]

def main():
    M_g = 1e10
    M = M_g * eVperg

    Ned = 2

    print(lifetime(M, Ned))

ti = 2 #integration start time (very early universe)
tf = 15e9 * speryear #integration end time (after today)
temp_t = np.logspace(np.log10(ti), np.log10(tf), 100000)
M_g = 1e10
M = M_g * eVperg
Mstar  = 1e4*1e9

Ned = 2
MBHoft = calcMoft(temp_t, M, Ned = Ned, Mstar = Mstar, odeMethod = 'BDF')
print(MBHoft)
plt.loglog(temp_t, MBHoft/eVperg)
plt.show()

