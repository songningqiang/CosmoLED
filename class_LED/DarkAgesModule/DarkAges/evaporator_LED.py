u"""
.. module:: evaporator
   :synopsis: Definition of functions in the scope of evaporating primordial black holes.
.. moduleauthor:: Patrick Stoecker <stoecker@physik.rwth-aachen.de>

This module contains the definition of functions which are useful for the calculation
of the evaporation of an evaporating black holes. This are mainly the following
functions:

#. Functions to translate between the mass of a black hole and its temperature

	* :meth:`get_mass_from_temperature <DarkAges.evaporator.get_mass_from_temperature>`
	  and :meth:`get_temperature_from_mass <DarkAges.evaporator.get_temperature_from_mass>`

#. Functions related to the spectrum of the particles produced by the evaporation:

	* :meth:`PBH_spectrum_at_m <DarkAges.evaporator.PBH_spectrum_at_m>`
	  and :meth:`PBH_primary_spectrum <DarkAges.evaporator.PBH_primary_spectrum>`
	* :meth:`PBH_F_of_M <DarkAges.evaporator.PBH_F_of_M>`
	  and :meth:`PBH_fraction_at_M <DarkAges.evaporator.PBH_fraction_at_M>`

#. Functions related to the evolution of the mass of the primordial black holes
   with time:

	* :meth:`PBH_dMdt <DarkAges.evaporator.PBH_dMdt>`
	* :meth:`PBH_mass_at_z <DarkAges.evaporator.PBH_mass_at_z>`

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from .common import time_at_z, logConversion, H, nan_clean
from .__init__ import DarkAgesError, get_redshift
from scipy.integrate import odeint as solve_ode
from scipy.integrate import trapz
from scipy.integrate import quad
from scipy.special import gamma
from scipy.interpolate import interp1d
import os
import sys

data_dir = os.path.join( os.path.dirname(os.path.realpath( __file__ )), 'data' )

#define constants
threshold_maximum = 200. #define the maximum ratio of E/T_BH to avoid overflow
gramtoGeV = 5.62e23
hbar = 6.582e-25 #GeV*s
Mpl = 1.22e19 #4d Planck mass in GeV


#particle_dict has structure 'particle_name':[mass,spin,dof,sigmoid_factor]
#neutrino is the sum of all flavors, particle/antiparticle are separated
#the names of gamma, w-boson and z-boson have been changed to match the keys in the Cirelli tables
particle_dict = {'photon': [0,1.0,2,0],
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
			   'wboson': [80.39,1.0,3,0],
			   'zboson': [91.19,1.0,3,0],
				 'gluon': [6e-1,1.0,16,1],
				 'higgs': [125.09,0.0,1,0],
				   'pi0': [1.350e-1,0.0,1,-1],
				  'piCh': [1.396e-1,0.0,1,-1]
}

def _particle_list_resolver( *particles ):
	particles = list(particles)
	if ('ALL' in particles) or ('all' in particles):
		return list(particle_dict.keys())
	else:
		if ('light quarks' in particles):
			particles.pop(particles.index('light quarks'))
			particles.extend(['up','down','strange'])
		if ('pions' in particles):
			particles.pop(particles.index('pions'))
			particles.extend(['pi0','piCh'])
		return particles

def _get_spin(particle):
	spin = particle_dict.get(particle)[1]
	return spin


def kn(n):
	#BlackMax convention
	#return (2.**n*np.pi**((n-3.)/2.)*gamma((n+3.)/2.)/(n+2.))**(1./(n+1.))
	#Dimopolos convention
	return (8.*np.pi**(-(n+1.)/2.)*gamma((n+3.)/2.)/(n+2.))**(1./(n+1.))

def get_radius_from_mass(mass,n,Mstar):
	#mass is in gram
	M = mass*gramtoGeV
	return kn(n)/Mstar*(M/Mstar)**(1./(n+1.))

def get_temperature_from_mass(mass,n,Mstar):
	u"""Returns the temperature of a black hole of a given mass

	Parameters
	----------
	mass : :obj:`float`
		Mass of the black hole (*in units of* :math:`\\mathrm{g}`)
	n : number of extra dimensions
	Mstar : Bulk Planck scale in GeV

	Returns
	-------
	:obj:`float`
		Temperature of the black hole (*in units of* :math:`\\mathrm{GeV}`)
	"""
	return (n+1.)/(4.*np.pi*get_radius_from_mass(mass,n,Mstar))

def get_mass_from_temperature(T_BH,n,Mstar):
	return Mstar*((n+1.)/(4.*np.pi*kn(n))*Mstar/T_BH)**(n+1)/gramtoGeV

#sigma/rh**2
def greybody_interp(spin, n):
	greybody_dir  = os.path.join(data_dir, 'greybodytables/')
	if spin == 0.0:
		data = np.loadtxt(greybody_dir+'greybody_scalar_n'+str(n)+'.txt')
	elif spin == 0.5:
		data = np.loadtxt(greybody_dir+'greybody_fermion_n'+str(n)+'.txt')
	elif spin == 1.0:
		data = np.loadtxt(greybody_dir+'greybody_gaugeboson_n'+str(n)+'.txt')
	else:
		raise TypeError('The spin "{:1.1}" is not recognized'.format(spin))
	f = interp1d(data[:,0], np.pi*data[:,1], kind = 'linear', bounds_error = False, fill_value = (0., np.pi*data[-1,1]))
	return f

#a nested dictionary to store the interpolation functions, speeding up the code 
greybodydict = {'scalar':{}, 'fermion':{}, 'gaugeboson':{}}
for n in range(0,7):
	greybodydict['scalar'][n] = greybody_interp(0.0, n)
	greybodydict['fermion'][n] = greybody_interp(0.5, n)
	greybodydict['gaugeboson'][n] = greybody_interp(1.0, n)

def QCD_factor(T_BH, sigmoid_factor):
	Lam_QCD = 0.3
	sigma = 0.1
	return 1. / (1. + np.exp(- sigmoid_factor*(np.log10(T_BH) - np.log10(Lam_QCD))/sigma))


def PBH_primary_spectrum( energy, PBH_mass, n, Mstar, ptype, **DarkOptions):
	u"""Returns the double differential spectrum, which does NOT sum up particles and antiparticles
	:math:`\\frac{\\mathrm{d}^2 N}{\\mathrm{d}E \\mathrm{d}t}` of particles with
	a given :code:`spin` and kinetic :code:`energy` for an evaporating black hole
	of mass :code:`PBH_mass`. Antiparticle included.

	Parameters
	----------
	energy : :obj:`float`
		Kinetic energy of the particle (*in units of* :math:`\\mathrm{GeV}`)
		produced by the evaporating black hole.
	PBH_mass : :obj:`float`
		Current mass of the black hole (*in units of* :math:`\\mathrm{g}`)
	ptype : particle name in particle_dict
	n : number of extra dimensions
	Mstar : Bulk Planck scale in GeV	

	Returns
	-------
	:obj:`float`
		Value of :math:`\\frac{\\mathrm{d}^2 N}{\\mathrm{d}E \\mathrm{d}t}`
		no unit
	"""


	if abs(PBH_mass) < np.inf and PBH_mass > 0. and energy > 0.:
		PBH_temperature = get_temperature_from_mass(PBH_mass,n,Mstar)
		if ptype in particle_dict.keys():
			pmass = particle_dict.get(ptype)[0]
			spin = particle_dict.get(ptype)[1]
			dof = particle_dict.get(ptype)[2]
			sigmoid_factor = particle_dict.get(ptype)[3]
			#don't include antiparticle, multiply the positron flux by a factor of 2 instead to accout for electrons
			#if spin == 0.5 or ptype == 'wboson' or ptype == 'piCh':
			#	dof *= 2.			
		else:
			raise TypeError('The particle {} is not recognized'.format(ptype))
		rh = get_radius_from_mass(PBH_mass,n,Mstar)
		if spin == 0.0:
			f = greybodydict['scalar'][n]
		elif spin == 0.5:
			f = greybodydict['fermion'][n]
		elif spin == 1.0:
			f = greybodydict['gaugeboson'][n]
		else:
			raise TypeError('The spin "{:1.1}" is not recognized'.format(spin))		
		Gamma = f(energy*rh)*rh**2*dof
		if sigmoid_factor != 0:
			Gamma = Gamma*QCD_factor(PBH_temperature,sigmoid_factor)
		if energy <= threshold_maximum*PBH_temperature and energy >= pmass:
			return (1/(2*np.pi**2))*Gamma*energy*np.sqrt(energy**2-pmass**2)/(np.exp(energy/PBH_temperature)-(-1)**int(np.round(2*spin)))
		else:
			return 0.
	else:
		return 0.


def PBH_energy_spectrum( energy, PBH_mass, ptype, n, Mstar):
	u"""Returns the double differential spectrum, which sums up particles and antiparticles
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

	if abs(PBH_mass) < np.inf and PBH_mass > 0. and energy > 0.:
		PBH_temperature = get_temperature_from_mass(PBH_mass,n,Mstar)
		if ptype in particle_dict.keys():
			pmass = particle_dict.get(ptype)[0]
			spin = particle_dict.get(ptype)[1]
			dof = particle_dict.get(ptype)[2]
			sigmoid_factor = particle_dict.get(ptype)[3]
			if spin == 0.5 or ptype == 'wboson' or ptype == 'piCh':
				dof *= 2.
		else:
			raise TypeError('The particle {} is not recognized'.format(ptype))
		rh = get_radius_from_mass(PBH_mass,n,Mstar)
		if spin == 0.0:
			f = greybodydict['scalar'][n]
		elif spin == 0.5:
			f = greybodydict['fermion'][n]
		elif spin == 1.0:
			f = greybodydict['gaugeboson'][n]
		else:
			raise TypeError('The spin "{:1.1}" is not recognized'.format(spin))		
		Gamma = f(energy*rh)*rh**2*dof	
		if sigmoid_factor != 0:
			Gamma = Gamma*QCD_factor(PBH_temperature,sigmoid_factor)
		if energy <= threshold_maximum*PBH_temperature and energy >= pmass:
			return (1/(2*np.pi**2))*Gamma*energy**2*np.sqrt(energy**2-pmass**2)/(np.exp(energy/PBH_temperature)-(-1)**int(np.round(2*spin)))
		else:
			return 0.
	else:
		return 0.		


def PBH_spectrum_at_m( mass, logEnergies, n, Mstar, *particles, **DarkOptions):
	u"""Returns the (combined) spectrum :math:`\\frac{\\mathrm{d}N}{\\mathrm{d}E}` of
	the particles given by the list :code:`*particles` emmited by a
	a black hole of mass :code:`mass` with a kinetic energy given by
	:code:`10**logEnergies`.

	This function computes for every particle the primary spectrum by
	:meth:`PBH_primary_spectrum <DarkAges.evaporator.PBH_primary_spectrum>` and the
	realtive contribution to the degrees of freedom :math:`\\mathcal{F}(M)` by
	:meth:`PBH_fraction_at_M <DarkAges.evaporator.PBH_fraction_at_M>` and computes the
	total spectrum.

	Parameters
	----------
	mass : :obj:`array-like`
		Array (:code:`shape = (k)`) of the black hole masses (*in units of* :math:`\\mathrm{g}`)
	logEnergies : :obj:`array-like`
		Array (:code:`shape = (l)`) of the logarithms (to the base 10) of the
		kinetic energy of the particles in question (the energy is given in units of GeV).
	*particles : tuple of :obj:`str`
		List of particles which should be considered. The contributions for each particle
		will be summed up. **Must at least contain one entry**

	Raises
	------
	DarkAgesError
		If no particles are given, (:code:`*particles` is empty)

	Returns
	-------
	:obj:`array-like`
		#Array (:code:`shape = (k,l)`) of the summed particle spectrum (*in units of* :math:`\\mathrm{GeV}^{-1}`)
		Array (:code:`shape = (k,l)`) of the summed particle spectrum No Unit
		at the enrgies and masses given by the inputs.
	"""

	ret = np.zeros((len(logEnergies),len(mass)), dtype=np.float64)
	E = logConversion(logEnergies - 9) #this converts energy from eV to GeV

	if particles is not None:
		particles = _particle_list_resolver( *particles )
		for particle in particles:
			#energy = E
			#kinetic energy + mass
			energy = E + particle_dict.get(particle)[0]*np.ones_like(E)
			ret[:,:] += np.asarray(np.vectorize(PBH_primary_spectrum, excluded = ['ptype','n','Mstar']).__call__(energy[:,None], mass[None,:], n, Mstar, particle, **DarkOptions))
	else:
		raise DarkAgesError('There is no particle given')
	return ret

#find integrated emission power
xi_scalar = np.array([0.0018695,0.0167,0.0675,0.1868,0.4156,0.8021,1.4011])
xi_fermion = np.array([0.001028,0.0146,0.0612,0.1666,0.3619,0.6836,1.1736])
xi_vector = np.array([4.2272e-04,0.0115,0.0611,0.1864,0.4316,0.8469,1.4884])
xi_graviton = np.array([9.655e-5,0.0097235,0.099484,0.4927,1.9042,6.8861,24.6844])
b_scalar = np.array([0.3952,0.3334,0.2832,0.2806,0.2846,0.2932,0.3044])
b_fermion = np.array([0.3370,0.2758,0.2933,0.2879,0.2895,0.2959,0.2739])
b_vector = np.array([0.2760,0.2203,0.2638,0.2743,0.2583,0.2975,0.3105])
c_scalar = np.array([1.186,1.236,1.291,1.296,1.292,1.282,1.27])
c_fermion = np.array([1.221,1.297,1.279,1.286,1.284,1.276,1.303])
c_vector = np.array([1.264,1.361,1.311,1.303,1.329,1.279,1.265])

def PBH_dMdt(PBH_mass, time, n, Mstar, scale=1,  **DarkOptions):
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
		ret = 0.
		rh = get_radius_from_mass(PBH_mass, n, Mstar)
		TH = (n+1.)/(4.*np.pi*rh)
		for ptype in particle_dict.keys():
			pmass = particle_dict.get(ptype)[0]
			spin = particle_dict.get(ptype)[1]
			dof = particle_dict.get(ptype)[2]
			sigmoid_factor = particle_dict.get(ptype)[3]
			#add particle and antiparticle
			if spin == 0.5 or ptype == 'wboson' or ptype == 'piCh':
				dof *= 2.

			if spin == 0.0:
				xi = xi_scalar[n]
				b = b_scalar[n]
				c = c_scalar[n]
			elif spin == 0.5:
				xi = xi_fermion[n]
				b = b_fermion[n]
				c = c_fermion[n]
			elif spin == 1.0:
				xi = xi_vector[n]
				b = b_vector[n]
				c = c_vector[n]
			else:
				raise TypeError('The spin "{:1.1}" is not recognized'.format(spin))

			if sigmoid_factor != 0:
				ret += dof*xi*np.exp(-b*(pmass/TH)**c)*QCD_factor(TH,sigmoid_factor)
			else:
				ret += dof*xi*np.exp(-b*(pmass/TH)**c)	

		#add gravitons
		ret += xi_graviton[n]
		ret = ret/(2.*np.pi)/rh**2

		gramtoGeV = 5.62e23
		hbar = 6.582e-25 #GeV*s
		return -ret/gramtoGeV/hbar
	else:
		return -0.0	

def PBH_mass_at_z(initial_PBH_mass, n, Mstar, redshift=None , **DarkOptions):
	u"""Solves the ODE for the PBH mass (:meth:`PBH_dMdt <DarkAges.evaporator.PBH_dMdt>`)
	and returns the masses at the redshifts given by the input :code:`redshift`

	If not specified by an additional keyword-argument in :code:`**DarkOptions`
	(by :code:`Start_evolution_at = ...`) the evolution is started at a redshift of 10.000

	Parameters
	----------
	initial_PBH_mass : :obj:`float`
		Initial mass of the primordial black hole (*in units of g*)
	redshift : :obj:`array-like` *optional*
		Array (:code:`shape = (l)`) of redshifts at which the PBH mass should
		be returned. If not given, the global redshift-array from
		:class:`the initializer <DarkAges.__init__>` is taken

	Returns
	-------
	:obj:`array-like`
		Array (:code:`shape = (l)`) of the PBH mass at the redshifts given in t
	"""

	# Jacobian of the ODE for the PBH mass.
	# Needed for better performance of the ODE-solver.
	#def jac(mass, time, n, Mstar, scale=1, **DarkOptions):
	#	out = np.ones((1,1))*(-2./((n+1)*mass))*PBH_dMdt(mass, time, n, Mstar, scale=scale , **DarkOptions) # partial_(dMdt) / partial_M
	#	#out = np.zeros((1,1))
	#	return out

	def jac(mass, time, n, Mstar, scale=1, **DarkOptions):
		out = np.ones((1,1))*(-2./((n+1)*mass))*PBH_dMdt(mass*scale, time, n, Mstar)/scale # partial_(dMdt) / partial_M
		#out = np.zeros((1,1))
		return out	

	if redshift is None:
		redshift = get_redshift()

	z_start = DarkOptions.get('Start_evolution_at', 1e4)

	if np.max(redshift) >= z_start:
		raise DarkAgesError("The redshift array is in conflict with the redshift at which to start the evolution of the balck hole. At least one entry in 'redshift' exceeds the value of 'z_start'")

	log10_ini_mass = np.log10(initial_PBH_mass)
	scale = 10**(np.floor(log10_ini_mass)+5)
	initial_PBH_mass *= 1/scale

	# Workaround for dealing with **DarkOptions inside the ODE-solver.
	#ODE_to_solve = lambda m,t: PBH_dMdt(m, t, n, Mstar, scale=scale, **DarkOptions)
	ODE_to_solve = lambda m,t: PBH_dMdt(m*scale, t, n, Mstar, scale=scale, **DarkOptions)/scale
	jac_to_use = lambda m,t: jac(m,t, n, Mstar, scale=scale, **DarkOptions)    

	#temp_t = 10**np.linspace(np.log10(time_at_z(z_start)), np.log10(time_at_z(1.)), 1e5)
	temp_t = 10**np.linspace(np.log10(time_at_z(z_start)), np.log10(time_at_z(np.min(redshift))), int(1e5))
	temp_mass, full_info = solve_ode(ODE_to_solve, initial_PBH_mass, temp_t, Dfun=jac_to_use, full_output=1,mxstep=10000)

	#dout = np.array([temp_t,temp_mass[:,0]])
	#np.savetxt('testout.txt',dout.transpose())	

	times =  time_at_z(redshift)
	PBH_mass_at_t = np.interp(times, temp_t, temp_mass[:,0])

	out = np.array([redshift, scale*PBH_mass_at_t])
	mask = out[1,:] <= 0.
	out[1,mask] = 0.

	return out
