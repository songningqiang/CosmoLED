from __future__ import absolute_import, division, print_function

import numpy as np
import math
import os
import sys
import dill
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import odeint as solve_ode
from scipy.integrate import ode
import mpmath as mp
import scipy.special as bessel
import matplotlib.pyplot as plt
import warnings


threshold_maximum = 100.#maximum emission is 100*T_BH

#define constants
gramtoGeV = 5.62e23
hbar = 6.582e-25 #GeV*s
hbarc = 0.197e-13 #GeV*cm
Mpl = 1.220910e19 #GeV
facc = 1. #accretion efficiency


def kn(n):
	#return (2.**n*math.pi**((n-3.)/2.)*math.gamma((n+3.)/2.)/(n+2.))**(1./(n+1.))
	return (8.*math.pi**(-(n+1.)/2.)*math.gamma((n+3.)/2.)/(n+2.))**(1./(n+1.))

def get_radius_from_mass(PBH_mass,n,Mstar):
	#mass is in gram
	gramtoGeV = 5.62e23
	M = PBH_mass*gramtoGeV
	return kn(n)/Mstar*(M/Mstar)**(1./(n+1.))

def get_temperature_from_mass(PBH_mass,n,Mstar):
	return (n+1.)/(4.*math.pi*get_radius_from_mass(PBH_mass,n,Mstar))

def get_mass_from_temperature(T_BH,n,Mstar):
	gramtoGeV = 5.62e23
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

#find the threshold for approximation and asymptotic value
def wrhthresh(n):
	return math.sqrt(3.)/4.*(n+1.)*((n+3.)/2.)**(1./(n+1.))*math.sqrt((n+3.)/(n+1.))*math.gamma(3./(n+1.))/math.gamma(1./(n+1))/math.gamma(2./(n+1))

#sigma/rh**2
def greybody_interp(spin, n):
	if spin == 0.0:
		data = np.loadtxt('greybodytables/'+'greybody_scalar_n'+str(n)+'.txt')
	elif spin == 0.5:
		data = np.loadtxt('greybodytables/'+'greybody_fermion_n'+str(n)+'.txt')
	elif spin == 1.0:
		data = np.loadtxt('greybodytables/'+'greybody_gaugeboson_n'+str(n)+'.txt')
	else:
		raise TypeError('The spin "{:1.1}" is not recognized'.format(spin))
	f = interp1d(data[:,0], math.pi*data[:,1], kind = 'linear', bounds_error = False, fill_value = (0., math.pi*data[-1,1]))
	return f

#a nested dictionary to store the interpolation functions, speeding up the code 
greybodydict = {'scalar':{}, 'fermion':{}, 'gaugeboson':{}}
for n in range(1,7):
	greybodydict['scalar'][n] = greybody_interp(0.0, n)
	greybodydict['fermion'][n] = greybody_interp(0.5, n)
	greybodydict['gaugeboson'][n] = greybody_interp(1.0, n)

#sigma/rh**2
def greybody_factor(wrh,spin,n):
	if spin == 0.5 or spin == 0.0:
		return math.pi*((n+3.)/2.)**(2./(n+1.))*(n+3.)/(n+1.)
	elif spin == 1.0:
		if wrh < wrhthresh(n):
			return 16.*math.pi/3./(n+1.)**2*wrh**2*(math.gamma(1./(n+1.))*math.gamma(2./(n+1.))/math.gamma(3./(n+1.)))**2
		else:
			return math.pi*((n+3.)/2.)**(2./(n+1.))*(n+3.)/(n+1.)
	else:
		#raise DarkAgesError('The spin "{:1.1%}" is not recognized'.format(spin))
		raise TypeError('The spin "{:1.1}" is not recognized'.format(spin))


def QCD_factor(T_BH, sigmoid_factor):
	Lam_QCD = 0.3
	sigma = 0.1
	return 1. / (1. + np.exp(- sigmoid_factor*(np.log10(T_BH) - np.log10(Lam_QCD))/sigma))

def PBH_primary_spectrum( energy, PBH_mass, ptype, n, Mstar):
	u"""Returns the double differential spectrum
	:math:`\\frac{\\mathrm{d}^2 N}{\\mathrm{d}E \\mathrm{d}t}` of particles with
	a given :code:`spin` and kinetic :code:`energy` for an evaporating black hole
	of mass :code:`PBH_mass`. Antiparticle not included.

	Parameters
	----------
	energy : :obj:`float`
		Kinetic energy of the particle (*in units of* :math:`\\mathrm{GeV}`)
		produced by the evaporating black hole.
	PBH_mass : :obj:`float`
		Current mass of the black hole (*in units of* :math:`\\mathrm{g}`)
	spin : :obj:`float`
		Spin of the particle produced by the evaporating black hole (Needs
		to be a multiple of :math:`\\frac{1}{2}`, i.e. :code:`2 * spin` is assumed
		to have integer value)

	Returns
	-------
	:obj:`float`
		Value of :math:`\\frac{\\mathrm{d}^2 N}{\\mathrm{d}E \\mathrm{d}t}`
	"""

	#threshold_maximum = 10.#maximum emission is 10*T_BH

	if abs(PBH_mass) < np.inf and PBH_mass > 0. and energy > 0.:
		PBH_temperature = get_temperature_from_mass(PBH_mass,n,Mstar)
		if ptype in particle_dict.keys():
			pmass = particle_dict.get(ptype)[0]
			spin = particle_dict.get(ptype)[1]
			dof = particle_dict.get(ptype)[2]
			sigmoid_factor = particle_dict.get(ptype)[3]
		else:
			raise TypeError('The particle {} is not recognized'.format(ptype))
		rh = get_radius_from_mass(PBH_mass,n,Mstar)
		#f = greybody_interp(spin,n)
		if spin == 0.0:
			f = greybodydict['scalar'][n]
		elif spin == 0.5:
			f = greybodydict['fermion'][n]
		elif spin == 1.0:
			f = greybodydict['gaugeboson'][n]
		else:
			raise TypeError('The spin "{:1.1}" is not recognized'.format(spin))
		Gamma = f(energy*rh)*rh**2*dof
		#Gamma = greybody_factor(energy*rh,spin,n)*rh**2*dof
		if sigmoid_factor != 0:
			Gamma = Gamma*QCD_factor(PBH_temperature,sigmoid_factor)
		if energy <= threshold_maximum*PBH_temperature and energy >= pmass:
			return (1/(2*np.pi**2))*Gamma*energy*np.sqrt(energy**2-pmass**2)/(np.exp(energy/PBH_temperature)-(-1)**int(np.round(2*spin)))
		else:
			return 0.
	else:
		return 0.

def PBH_energy_spectrum( energy, PBH_mass, ptype, n, Mstar):
	u"""Returns the double differential spectrum
	:math:`\\frac{\\mathrm{d}^2 E}{\\mathrm{d}energy \\mathrm{d}t}` of particles with
	a given :code:`spin` and kinetic :code:`energy` for an evaporating black hole
	of mass :code:`PBH_mass`. Both particle and antiparticle are included.

	Parameters
	----------
	energy : :obj:`float`
		Kinetic energy of the particle (*in units of* :math:`\\mathrm{GeV}`)
		produced by the evaporating black hole.
	PBH_mass : :obj:`float`
		Current mass of the black hole (*in units of* :math:`\\mathrm{g}`)
	spin : :obj:`float`
		Spin of the particle produced by the evaporating black hole (Needs
		to be a multiple of :math:`\\frac{1}{2}`, i.e. :code:`2 * spin` is assumed
		to have integer value)

	Returns
	-------
	:obj:`float`
		Value of :math:`\\frac{\\mathrm{d}^2 N}{\\mathrm{d}E \\mathrm{d}t}`
	"""

	#threshold_maximum = 10.#maximum emission is 10*T_BH

	if abs(PBH_mass) < np.inf and PBH_mass > 0. and energy > 0.:
		PBH_temperature = get_temperature_from_mass(PBH_mass,n,Mstar)
		if ptype in particle_dict.keys():
			pmass = particle_dict.get(ptype)[0]
			spin = particle_dict.get(ptype)[1]
			dof = particle_dict.get(ptype)[2]
			sigmoid_factor = particle_dict.get(ptype)[3]
			if spin == 0.5 or ptype == 'w-boson' or ptype == 'piCh':
				dof *= 2.
		else:
			raise TypeError('The particle {} is not recognized'.format(ptype))
		rh = get_radius_from_mass(PBH_mass,n,Mstar)
		#f = greybody_interp(spin,n)
		if spin == 0.0:
			f = greybodydict['scalar'][n]
		elif spin == 0.5:
			f = greybodydict['fermion'][n]
		elif spin == 1.0:
			f = greybodydict['gaugeboson'][n]
		else:
			raise TypeError('The spin "{:1.1}" is not recognized'.format(spin))		
		Gamma = f(energy*rh)*rh**2*dof		
		#Gamma = greybody_factor(energy*rh,spin,n)*rh**2*dof
		if sigmoid_factor != 0:
			Gamma = Gamma*QCD_factor(PBH_temperature,sigmoid_factor)
		if energy <= threshold_maximum*PBH_temperature and energy >= pmass:
			return (1/(2*np.pi**2))*Gamma*energy**2*np.sqrt(energy**2-pmass**2)/(np.exp(energy/PBH_temperature)-(-1)**int(np.round(2*spin)))
		else:
			return 0.
	else:
		return 0.		

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
		ret = ret/(2.*np.pi)/rh**2

		return -ret/gramtoGeV/hbar
	else:
		return -0.0			

#columns: temperature (eV), gstar entropy, gstar energy density
dgstar = np.loadtxt('gstar.txt')
fgstar = interp1d(np.log10(dgstar[:,0]) - 9., dgstar[:,2], kind = 'linear', bounds_error = False, fill_value = (dgstar[0,2], dgstar[-1,2]))

#"""
#gstar by Aaron
def gstarofT(T, sflag = False):
	#T: temperature in GeV
	#sflag: if true, use gstars; if false, use gstar not s
	#returns gstar as computed by the interpolation in http://arxiv.org/pdf/0910.1066.pdf

	if not sflag:
		a0 = 1.21;
		a1 = np.array([ 0.572,   0.330,   0.579,   0.138,   0.108])
		a2 = np.array([ -8.77,   -2.95,   -1.80,   -0.162,   3.76])
		a3 = np.array([ 0.682,   1.01,   0.165,   0.934,   0.869])
		T = np.log(np.array([T, T, T, T, T]))
		Imat = np.ones(5)
		gstar = a0 + np.sum(a1*(Imat+np.tanh((T-a2)/a3)))
	else:
		aS0 =1.36
		aS1 = np.array([ 0.498,   0.327,   0.579,   0.140,   0.109])
		aS2 = np.array([ -8.74,   -2.89,   -1.79,   -0.102,   3.82])
		aS3 = np.array([ 0.693,   1.01,   0.155,   0.963,   0.907])
		T = np.log(np.array([T, T, T, T, T]))
		Imat = np.ones(5)
		gstar = aS0 + np.sum(aS1*(Imat+np.tanh((T-aS2)/aS3)))

	gstar = np.exp(gstar)
	return gstar
#"""

#gstar by Aaron, less efficent but could be used in findroot
def gstarofT_mp(T, sflag = False):
	#T: temperature in GeV
	#sflag: if true, use gstars; if false, use gstar not s
	#returns gstar as computed by the interpolation in http://arxiv.org/pdf/0910.1066.pdf

	if not sflag:
		a0 = 1.21;
		a1 = np.array([ 0.572,   0.330,   0.579,   0.138,   0.108])
		a2 = np.array([ -8.77,   -2.95,   -1.80,   -0.162,   3.76])
		a3 = np.array([ 0.682,   1.01,   0.165,   0.934,   0.869])
		T = np.ones(5)*mp.log(T)
		Imat = np.ones(5)
		gstar = a0
		for idx in np.arange(5):
			gstar += a1[idx]*(Imat[idx]+mp.tanh((T[idx]-a2[idx])/a3[idx]))
	else:
		aS0 =1.36
		aS1 = np.array([ 0.498,   0.327,   0.579,   0.140,   0.109])
		aS2 = np.array([ -8.74,   -2.89,   -1.79,   -0.102,   3.82])
		aS3 = np.array([ 0.693,   1.01,   0.155,   0.963,   0.907])
		T = np.ones(5)*mp.log(T)
		Imat = np.ones(5)
		gstar = aS0
		for idx in np.arange(5):
			gstar += aS1[idx]*(Imat[idx]+mp.tanh((T[idx]-aS2[idx])/aS3[idx]))

	gstar = mp.exp(gstar)
	return gstar

def getalpha(MBH, n, Mstar):
	#return dimensionless factor alpha for evaporation
	#MBH is BH mass in GeV
	PBH_mass = MBH/gramtoGeV
	TH = get_temperature_from_mass(PBH_mass, n, Mstar)
	return -PBH_dMdt(PBH_mass, 0, n, Mstar)*gramtoGeV*hbar/TH**2

def getbeta(T, n, fgstar = gstarofT):
#def getbeta(T, n, fgstar = fgstar):
	gs = fgstar(T)
	return np.pi/120.*(n+1)**2*facc*gs

def dMdt(MBH, T, n, Mstar, evaporationflag = False):
	#return dM/dt in GeV^2, this function is to avoid divergence when MBH=0
	#MBH is BH mass in GeV
	#if turn on evaporationflag, BH evaporation is allowed, otherwise  just consider accretion
	if MBH > 0:
		PBH_mass = MBH/gramtoGeV
		TH = get_temperature_from_mass(PBH_mass, n, Mstar)			
		if evaporationflag:	
			alpha = getalpha(MBH, n, Mstar)
			beta = getbeta(T, n)
			ret = (-alpha+beta*T**4/TH**4)*TH**2
		else:
			beta = getbeta(T, n)
			ret = beta*T**4/TH**2
	else:
		ret = 0.

	return ret

def PBH_dMdT_RD(T, PBH_mass, fgstar, n, Mstar):
	#PBH_mass is in gram
	#return dimensionless dM/dT assuming radiation domination
	TH = get_temperature_from_mass(PBH_mass, n, Mstar)
	PBH_mass = PBH_mass
	alpha = -PBH_dMdt(PBH_mass, 0, n, Mstar)*gramtoGeV*hbar/TH**2
	#alpha = 0. #for test, remove evaporation
	gs = fgstar(T)
	beta = np.pi/120.*(n+1)**2*facc*gs
	#beta = 0. #for test, remove accretion
	#print(alpha, beta)
	if PBH_mass > 0:
		return -np.sqrt(45./gs/4/math.pi**3)*Mpl*TH**2/T**3*(-alpha+beta*T**4/TH**4)
	else:
		return -0.

def PBH_dMdlgT(lgT, PBH_mass, fgstar, n, Mstar):
	T = 10.**lgT
	print(lgT, PBH_mass, PBH_dMdT(T, PBH_mass, fgstar, n, Mstar)*T*np.log(10.))
	return PBH_dMdT(T, PBH_mass, fgstar, n, Mstar)*T*np.log(10.)

def dTdt_RD(T):
	#return dTdt in GeV^2, this works only if radiation dominates
	gs = fgstar(T)
	return np.sqrt(4*np.pi**3/45*gs)*T**3/Mpl

def gamman(T_ini, n):
	gs = fgstar(T_ini)
	return np.sqrt(np.pi**3/160*gs)*facc*kn(n)**2*(n-1.)/(n+1)

def Mas(T_ini, n, Mstar):
	#returns the asymptotic BH mass in GeV produced at T_ini
	return (gamman(T_ini, n)*Mpl*T_ini**2/Mstar**3)**((n+1.)/(n-1.))*Mstar

def guessTth(n, Mstar):
	TH = get_temperature_from_mass(Mstar/gramtoGeV, n, Mstar)
	alpha = -PBH_dMdt(Mstar/gramtoGeV, 0, n, Mstar)*gramtoGeV*hbar/TH**2
	gs = 106.75
	beta = np.pi/120.*(n+1)**2*facc*gs
	return (alpha/beta)**0.25*TH

def getTth(n, Mstar, fgstar = gstarofT_mp):
	#fint exact Tth when BH created at Mstar begins to decay
	Tguess = guessTth(n, Mstar)
	TH = get_temperature_from_mass(Mstar/gramtoGeV, n, Mstar)
	alpha = getalpha(Mstar, n, Mstar)
	ret = mp.findroot(lambda T: (alpha/getbeta(T, n, fgstar))**0.25*TH-T, Tguess)
	return float(ret)

#this function not exact
def guessMth(T, n, Mstar):
	#return the minimum BH mass when BH will accrete in GeV
	if T > getTth(n, Mstar):
		ret = Mstar
	else:
		alpha = getalpha(Mstar, n, Mstar)
		beta = getbeta(T, n)
		ret = ((n+1)/(4*np.pi*kn(n))*(alpha/beta)**0.25*Mstar/T)**(n+1)*Mstar

	return ret

"""
def getMth(T, n, Mstar, fgstar = gstarofT_mp):
	#return the minimum BH mass when BH will accrete in GeV
	if T > getTth(n, Mstar):
		ret = Mstar
	else:
		Mguess = guessMth(T, n, Mstar)
		ret = mp.findroot(lambda M: ((n+1)/(4*np.pi*kn(n))*(getalpha(M, n, Mstar)/getbeta(T, n, fgstar = fgstar))**0.25*Mstar/T)**(n+1)*Mstar-M, Mguess)

	return float(ret)
"""

def dGammadM(T, MBH, n, Mstar):
	#MBH is in GeV
	#this returns dGamma/dM in GeV^3
	gs = fgstar(T)
	return gs**2*kn(n)**2/16./np.pi**3*(MBH/Mpl)**((2*n+4.)/(n+1))*MBH*T*(T*bessel.kn(2,np.sqrt(2)*MBH/T)+np.sqrt(2)*MBH*bessel.kn(1,np.sqrt(2)*MBH/T))

def GammaTot(T, n, Mstar):
	#return int dGamma/dM dM in GeV^4, M0 is the minimum BH mass for accretion in GeV
	M0 = guessMth(T, n, Mstar)
	return (quad(lambda x: dGammadM(T, x, n, Mstar), M0, np.inf))[0]

def rhor(T, fgstar = gstarofT):
	#return radiation energy density in GeV^4
	return np.pi**2/30*fgstar(T)*T**4;

def rhoBH(Ts, MBH, hT):
	#return the energy density of BHs in GeV^4
	#Ts is a vector of plasma temperature for BH production in GeV
	#MBH is a vector of BH mass in GeV
	#hT is a vector of dn/dT in GeV^2
	#Ts, MBH, hT must be of the same size
	return np.abs(np.trapz(MBH*hT,Ts))

def H(T, Ts, MBH, hT):
	#return Hubble in GeV
	return np.sqrt(8*np.pi/3.*(rhor(T)+rhoBH(Ts, MBH, hT)))/Mpl

"""
def dTdt_abs(T, Ts, MBH, hT, fgstr, n, Mstar):
	#return |dT/dt| in GeV^2
	#T is the plasma temperature in the current step
	beta = getbeta(T, n)
	gs = fgstar(T)
	TBH = np.asarray(np.vectorize(get_temperature_from_mass, excluded = ['n','Mstar']).__call__(MBH/gramtoGeV, n, Mstar))
	return H(T, Ts, MBH, hT)*T+15*beta/(2*np.pi**2*gs)*T*np.abs(np.trapz(hT/TBH**2, Ts))
"""

#"""
def dTdt_abs(T, Ts, MBH, hT, fgstr, n, Mstar):
	#return |dT/dt| in GeV^2
	#T is the plasma temperature in the current step
	beta = getbeta(T, n)
	gs = fgstar(T)
	rho_r = rhor(T)
	dMdt_vec = np.asarray(np.vectorize(dMdt, excluded = ['T','n','Mstar','evaporationflag']).__call__(MBH, T, n, Mstar, evaporationflag = False))
	return H(T, Ts, MBH, hT)*T+T/(4*rho_r)*np.abs(np.trapz(hT*dMdt_vec, Ts))
#"""

def prodrate(T, Ts, MBH, hT, fgstr, n, Mstar):
	#return nt dGamma/dM dM / dT/dt in GeV^2
	return GammaTot(T, n, Mstar)/dTdt_abs(T, Ts, MBH, hT, fgstr, n, Mstar)

def dTdt(T, Ts, MBH, hT, fgstr, n, Mstar):
	return -dTdt_abs(T, Ts, MBH, hT, fgstr, n, Mstar)

def dTdt_abs_RD(T, fgstar = gstarofT):
	gs = fgstar(T)
	return np.sqrt(4*np.pi**3*gs/45)*T**3/Mpl

"""
def dMdT(T, Ts, MBH, hT, fgstr, n, Mstar):
	#return dimensionless dM/dT vector
	beta = getbeta(T, n)
	TBH = np.asarray(np.vectorize(get_temperature_from_mass, excluded = ['n','Mstar']).__call__(MBH/gramtoGeV, n, Mstar))
	dT = dTdt(T, Ts, MBH, hT, fgstr, n, Mstar)
	return beta*T**4/TBH**2/dT
"""

#"""
def dMdT(T, Ts, MBH, hT, fgstr, n, Mstar):
	#return dimensionless dM/dT vector
	dMdt_vec = np.asarray(np.vectorize(dMdt, excluded = ['T','n','Mstar','evaporationflag']).__call__(MBH, T, n, Mstar, evaporationflag = False))
	dT = dTdt(T, Ts, MBH, hT, fgstr, n, Mstar)
	return dMdt_vec/dT
#"""


def dhdT(T, Ts, MBH, hT, fgstr, n, Mstar):
	#return dh_T/dT vector in GeV
	dT = dTdt(T, Ts, MBH, hT, fgstr, n, Mstar)
	return -3*H(T, Ts, MBH, hT)*hT/dT


def hM_RD(T, n, Mstar):
	#return dndM in (cm^3*gram)^-1
	Tth = guessTth(n, Mstar)
	if T > Tth:
		gs = fgstar(T)
		return GammaTot(T, n, Mstar)/((np.sqrt(16*np.pi**3/45.*gs))*(n+1.)/(n-1)*Mas(T, n, Mstar)/Mpl*T**2)/hbarc**3*gramtoGeV
	else:
		return 0.

#def dndM_RD(TRH, Tend, n, Mstar):
#	gsRH = fgstar(TRH)
#	gsend = fgstar(Tend)



n = 2
Mstar = 1.e4
NT = 500 #number of T bins for BH production

print(getTth(n,Mstar))

"""
print(getalpha(1e20, n, Mstar))
sys.exit(0)

Tth = guessTth(n, Mstar)
T0 = Tth/2
Tend = Tth/10
Ts = np.logspace(np.log10(T0), np.log10(Tend), NT)
M0 = Mstar/gramtoGeV
M0GeV = guessMth(T0, n, Mstar)+1

ODE_to_solve = lambda T, y: np.concatenate([dMdT(T, Ts, y[0:NT], y[NT:2*NT], gstarofT, n, Mstar), dhdT(T, Ts, y[0:NT], y[NT:2*NT], gstarofT, n, Mstar)])
MBH = np.asarray(np.vectorize(guessMth, excluded = ['n', 'Mstar']).__call__(Ts, n, Mstar))
y0 = np.concatenate([MBH, np.array([GammaTot(Ts[0], n, Mstar)/dTdt_abs_RD(Ts[0])]), np.zeros(NT-1)])
#print(y0)

solver = ode(ODE_to_solve).set_integrator('vode', rtol = 1e-3)
solver.set_initial_value(y0, Ts[0])
y0 = solver.integrate(Ts[1])
print(y0)
sys.exit()

for i in np.arange(1, NT-1):
	print(i)
	y0[i:NT] = np.asarray(np.vectorize(guessMth, excluded = ['n', 'Mstar']).__call__(Ts[i:NT], n, Mstar))
	y0[NT+i] = prodrate(Ts[i], Ts, y0[0:NT], y0[NT:2*NT], gstarofT, n, Mstar)
	solver.set_initial_value(y0, Ts[i])
	y0 = solver.integrate(Ts[i+1])
y0[NT-1] = getMth(Ts[NT-1], n, Mstar)
y0[2*NT-1] = prodrate(Ts[NT-1], Ts, y0[0:NT], y0[NT:2*NT], gstarofT, n, Mstar)
print(Ts[NT-1], solver.t, y0)
"""

"""
#logscale does not work well for decay
ODE_to_solve = lambda lnT, lnM: np.exp(lnT)/np.exp(lnM)*PBH_dMdT(np.exp(lnT), np.exp(lnM)/gramtoGeV, gstar, n, Mstar)
#print(ODE_to_solve(np.log(1.e3), np.log(1.e7)))
solver = ode(ODE_to_solve).set_integrator('lsoda', rtol = 1.e-2, max_order_s = 2, max_order_ns = 4)
solver.set_initial_value(np.log(M0GeV), np.log(T0))
sol = []

dlnT = (np.log(Tend) - np.log(T0))/1000.
while solver.successful() and solver.t > np.log(Tend):
	solver.integrate(solver.t+dlnT, step = True)
	sol.append([solver.t, solver.y[0]])

sol = np.array(sol)
plt.loglog(np.exp(sol[:,0]), np.exp(sol[:,1]), 'b.-')
plt.show()
"""

#"""
T0 = 1.e4
Tend=1e-4 #BBN
M0GeV = 1.e+04
ODE_to_solve = lambda T, M: PBH_dMdT_RD(T, M/gramtoGeV, fgstar, n, Mstar)
#solver = ode(ODE_to_solve).set_integrator('lsoda', nsteps = 1, rtol = 1.e-4, max_order_s = 2, max_order_ns = 4)
solver = ode(ODE_to_solve).set_integrator('vode', nsteps = 1, rtol = 1.e-4, order = 2)
solver.set_initial_value(M0GeV, T0)
sol = []

dT = (Tend - T0)/1000.
#while solver.successful() and solver.t > Tend:
#	solver.integrate(solver.t+dT, step = True)
#	sol.append([solver.t, solver.y[0]])

Tend = Tend
while solver.successful() and solver.t > Tend:
	solver.integrate(Tend, step = True)
	sol.append([solver.t, solver.y[0]])

sol = np.array(sol)
print(sol)
plt.loglog(sol[:,0], sol[:,1]/gramtoGeV, 'b.-')
plt.show()
#"""


