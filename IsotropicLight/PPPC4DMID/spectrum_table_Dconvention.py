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
import matplotlib.pyplot as plt

#define constants
threshold_maximum = 100. #define the maximum ratio of E/T_BH to avoid overflow
gramtoGeV = 5.62e23
hbar = 6.582e-25 #GeV*s
Mpl = 1.22e19 #4d Planck mass in GeV


def kn(n):
	#return (2.**n*math.pi**((n-3.)/2.)*math.gamma((n+3.)/2.)/(n+2.))**(1./(n+1.))
	return (8.*math.pi**(-(n+1.)/2.)*math.gamma((n+3.)/2.)/(n+2.))**(1./(n+1.))

def get_radius_from_mass(mass,n,Mstar):
	#mass is in gram
	#if n=0, Mstar must be set to Mpl
	M = mass*gramtoGeV
	return kn(n)/Mstar*(M/Mstar)**(1./(n+1.))

def get_temperature_from_mass(mass,n,Mstar):
	return (n+1.)/(4.*math.pi*get_radius_from_mass(mass,n,Mstar))

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

"""
#find the threshold for approximation and asymptotic value
def wrhthresh(n):
	return math.sqrt(3.)/4.*(n+1.)*((n+3.)/2.)**(1./(n+1.))*math.sqrt((n+3.)/(n+1.))*math.gamma(3./(n+1.))/math.gamma(1./(n+1))/math.gamma(2./(n+1))

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
"""

#sigma/rh**2
def greybody_interp(spin, n):
	if spin == 0.0:
		data = np.loadtxt('PPPC4DMID/greybodytables/'+'greybody_scalar_n'+str(n)+'.txt')
	elif spin == 0.5:
		data = np.loadtxt('PPPC4DMID/greybodytables/'+'greybody_fermion_n'+str(n)+'.txt')
	elif spin == 1.0:
		data = np.loadtxt('PPPC4DMID/greybodytables/'+'greybody_gaugeboson_n'+str(n)+'.txt')
	else:
		raise TypeError('The spin "{:1.1}" is not recognized'.format(spin))
	f = interp1d(data[:,0], math.pi*data[:,1], kind = 'linear', bounds_error = False, fill_value = (0., math.pi*data[-1,1]))
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
		if energy >= pmass and energy <= threshold_maximum*PBH_temperature: #avoid overflow
			return (1/(2*np.pi**2))*Gamma*energy*np.sqrt(energy**2-pmass**2)/(np.exp(energy/PBH_temperature)-(-1)**int(np.round(2*spin)))
		else:
			return 0.
	else:
		return 0.

def PBH_energy_spectrum( energy, PBH_mass, ptype, n, Mstar):
	u"""Returns the double differential spectrum in unit of GeV
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

		return -ret/gramtoGeV/hbar
	else:
		return -0.0		
"""
def PBH_dMdt(PBH_mass, time, n, Mstar):

	if PBH_mass > 0:
		ret = 0.
		for ptype in particle_dict.keys():
			pmass = particle_dict.get(ptype)[0]
			TH = get_temperature_from_mass(PBH_mass, n, Mstar)
			#threshold_maximum = 10.#maximum emission is 10*T_BH
			Emax = TH * threshold_maximum
			if pmass < Emax:
				ret += (quad(lambda x: PBH_energy_spectrum(x, PBH_mass, ptype, n, Mstar), pmass, Emax))[0]

		#add gravitons
		phasespaceintegral = np.array([0.00972353, 0.099579, 0.492872, 1.90469, 6.88716, 24.6866])
		rh = get_radius_from_mass(PBH_mass,n,Mstar)
		ret += phasespaceintegral[n-1]/(2.*math.pi)/rh**2

		gramtoGeV = 5.62e23
		hbar = 6.582e-25 #GeV*s
		return -ret/gramtoGeV/hbar
	else:
		return -0.0
"""

column_dict = {
	'electron': 4,
	'muon': 7,
	'tau':	10,
	'quark':11,
	'charm':12,
	'bottom':13,
	'top':14,
	'w-boson':17,
	'z-boson':20,
	'gluon':21,
	'gamma':22,
	'higgs':23,
	'nue':24,
	'numu':25,
	'nutau':26
}

data_dir = os.path.dirname(os.path.realpath( __file__ ))

def get_cirelli_spectra(key):
	EW_cols = (0, 1, column_dict.get(key))

	data_elec_EW = np.genfromtxt(os.path.join(data_dir,'AtProduction_positrons.dat'), unpack=True, usecols=EW_cols, skip_header=1)
	data_phot_EW = np.genfromtxt(os.path.join(data_dir,'AtProduction_gammas.dat'), unpack=True, usecols=EW_cols, skip_header=1)
	data_nu_e_EW = np.genfromtxt(os.path.join(data_dir,'AtProduction_neutrinos_e.dat'), unpack=True, usecols=EW_cols, skip_header=1)
	data_nu_m_EW = np.genfromtxt(os.path.join(data_dir,'AtProduction_neutrinos_mu.dat'), unpack=True, usecols=EW_cols, skip_header=1)
	data_nu_t_EW = np.genfromtxt(os.path.join(data_dir,'AtProduction_neutrinos_tau.dat'), unpack=True, usecols=EW_cols, skip_header=1)
	data_prot_EW = np.genfromtxt(os.path.join(data_dir,'AtProduction_antiprotons.dat'), unpack=True, usecols=EW_cols, skip_header=1)
	data_deut_EW = np.genfromtxt(os.path.join(data_dir,'AtProduction_antideuterons.dat'), unpack=True, usecols=EW_cols, skip_header=1)

	masses = np.unique(data_elec_EW[0,:])
	log10X = np.unique(data_elec_EW[1,:])
	dim1 = len(masses)
	dim2 = len(log10X)
	dNdlog10X_el = data_elec_EW[2,:].reshape(dim1,dim2) #this considers only primary particle, not antiparticle
	dNdlog10X_ph = data_phot_EW[2,:].reshape(dim1,dim2) #this consdiers both particle and antiparticle
	dNdlog10X_nu = (data_nu_e_EW[2,:] + data_nu_m_EW[2,:] + data_nu_t_EW[2,:]).reshape(dim1,dim2)
	dNdlog10X_pr = (data_prot_EW[2,:]/2.).reshape(dim1,dim2)#don't assume neutron decay, proton=neutron
	dNdlog10X_de = data_deut_EW[2,:].reshape(dim1,dim2)

	return masses, log10X, dNdlog10X_el, dNdlog10X_ph, dNdlog10X_nu, dNdlog10X_pr, dNdlog10X_de


def interp_log(x, y, x0):
	f = interp1d(x, y, kind = 'linear', bounds_error = False, fill_value = (0.,0.))
	return f(x0)


def sample_spectrum(input_spec_el, input_spec_ph, input_spec_nu, input_spec_pr, input_spec_de, input_log10E, m, sampling_log10E):
	u"""Returns the interpolated and properly normalized particle spectrum

	This method interpolates the particle spectra defined at the points
	:code:`input_log10E`, applies the normalization given the injection history
	in question and returns the recurrent spectra ath the points given in
	:code:`sampling_log10E`

	Parameters
	----------
	input_spec_el : :obj:`array-like`
		Array (:code:`shape = (k)`) of the differential spectrum
		:math:`\\frac{\\mathrm{d}N}{\\mathrm{d}E}` of electrons and positrons.
	input_spec_ph : :obj:`array-like`
		Array (:code:`shape = (k)`) of the differential spectrum
		:math:`\\frac{\\mathrm{d}N}{\\mathrm{d}E}` of photons.
	input_spec_oth : :obj:`array-like`
		Array (:code:`shape = (k)`) of the differential spectrum
		:math:`\\frac{\\mathrm{d}N}{\\mathrm{d}E}` of particles not injecting
		energy into the IGM. Used for the proper normailzation of the spectra.
	input_log10E : :obj:`array-like`
		Array (:code:`shape = (k)`) of the logarithm of the kinetic energies
		of the particles to the base 10 at which the input spectra are
		defined.
	m : :obj:`float`
		Masss of the DM candidate.
	sampling_log10E : :obj:`array-like`
		Array (:code:`shape = (l)`) of the logarithm of the kinetic energies
		(*in units of* :math:`\\mathrm{eV}`) of the particles to the base 10
		at which the spectra should be interpolated.

	Returns
	-------
	:obj:`array-like`
		Array (:code:`shape = (3,l)`) of the sampled particle spectra of
		positrons and electrons, photons, and particle not interacting
		with the IGM at the energies specified in :code:`sampling_log10E`.
	"""

	out_el = interp_log(input_log10E, input_spec_el, sampling_log10E)
	out_ph = interp_log(input_log10E, input_spec_ph, sampling_log10E)
	out_nu = interp_log(input_log10E, input_spec_nu, sampling_log10E)
	out_pr = interp_log(input_log10E, input_spec_pr, sampling_log10E)
	out_de = interp_log(input_log10E, input_spec_de, sampling_log10E)

	return np.array([out_el, out_ph, out_nu, out_pr, out_de])

def secondaries_from_cirelli(logEnergies,mass,primary):
	#from .common import sample_spectrum
	#cirelli_dir = os.path.join(data_dir, 'cirelli')
	cirelli_dir = '.'
	dumpername = 'cirelli_spectrum_of_{:s}.obj'.format(primary)

	equivalent_mass = mass
	if equivalent_mass < 5 or equivalent_mass > 1e5:
		raise ValueError('The spectra of Cirelli are only given in the range [5 GeV, 1e2 TeV] assuming DM annihilation. The equivalent mass for the given injection_history ({:.2g} GeV) is not in that range.'.format(equivalent_mass))

	if not hasattr(logEnergies,'__len__'):
		logEnergies = np.asarray([logEnergies])
	else:
		logEnergies = np.asarray(logEnergies)

	if not os.path.isfile( os.path.join(cirelli_dir, dumpername)):
		#sys.path.insert(1,cirelli_dir)
		masses, log10X, dNdLog10X_el, dNdLog10X_ph, dNdLog10X_nu, dNdlog10X_pr, dNdlog10X_de = get_cirelli_spectra(primary)
		total_dNdLog10X = np.asarray([dNdLog10X_el, dNdLog10X_ph, dNdLog10X_nu, dNdlog10X_pr, dNdlog10X_de])
		from interpolator import NDlogInterpolator
		#import interpolator
		interpolator = NDlogInterpolator(masses, np.rollaxis(total_dNdLog10X,1), exponent = 0, scale = 'log-log')
		dump_dict = {'dNdLog10X_interpolator':interpolator, 'log10X':log10X}
		with open(os.path.join(cirelli_dir, dumpername),'wb') as dump_file:
			dill.dump(dump_dict, dump_file)
	else:
		with open(os.path.join(cirelli_dir, dumpername),'rb') as dump_file:
			dump_dict = dill.load(dump_file)
			interpolator = dump_dict.get('dNdLog10X_interpolator')
			log10X = dump_dict.get('log10X')
	del dump_dict
	temp_log10E = log10X + np.log10(equivalent_mass)*np.ones_like(log10X)
	temp_el, temp_ph, temp_nu, temp_pr, temp_de = interpolator.__call__(equivalent_mass)	
	#temp_el, temp_ph, temp_nu, temp_pr, temp_de = interpolator.__call__(equivalent_mass) / (10**temp_log10E * np.log(10))[None,:]
	ret_spectra = np.empty(shape=(5,len(logEnergies)))
	ret_spectra = sample_spectrum(temp_el, temp_ph, temp_nu, temp_pr, temp_de, temp_log10E, equivalent_mass, logEnergies)
	return ret_spectra


def get_decay_spectra(key):
	if not key in ['muon', 'pi0', 'piCh', 'tau']:
		raise err('Decay table only provided for muon, pi0, piCh and tau.')

	data_elec = np.genfromtxt(os.path.join(data_dir,'secondariesfrom'+key+'.dat'), unpack=True, usecols=(0, 1, 2), skip_header=1)
	data_phot = np.genfromtxt(os.path.join(data_dir,'secondariesfrom'+key+'.dat'), unpack=True, usecols=(0, 1, 3), skip_header=1)
	masses = np.unique(data_elec[0,:])
	log10X = np.unique(data_elec[1,:])
	dim1 = len(masses)
	dim2 = len(log10X)
	dNdlog10X_el = data_elec[2,:].reshape(dim1,dim2) #this considers only primary particle, not antiparticle
	dNdlog10X_ph = data_phot[2,:].reshape(dim1,dim2) #this consdiers both particle and antiparticle

	return masses, log10X, dNdlog10X_el, dNdlog10X_ph


def sample_spectrum_decay(input_spec_el, input_spec_ph, input_log10E, m, sampling_log10E):
	u"""Returns the interpolated and properly normalized particle spectrum

	This method interpolates the particle spectra defined at the points
	:code:`input_log10E`, applies the normalization given the injection history
	in question and returns the recurrent spectra ath the points given in
	:code:`sampling_log10E`

	Parameters
	----------
	input_spec_el : :obj:`array-like`
		Array (:code:`shape = (k)`) of the differential spectrum
		:math:`\\frac{\\mathrm{d}N}{\\mathrm{d}E}` of electrons and positrons.
	input_spec_ph : :obj:`array-like`
		Array (:code:`shape = (k)`) of the differential spectrum
		:math:`\\frac{\\mathrm{d}N}{\\mathrm{d}E}` of photons.
	input_log10E : :obj:`array-like`
		Array (:code:`shape = (k)`) of the logarithm of the kinetic energies
		of the particles to the base 10 at which the input spectra are
		defined.
	m : :obj:`float`
		Masss of the DM candidate.
	sampling_log10E : :obj:`array-like`
		Array (:code:`shape = (l)`) of the logarithm of the kinetic energies
		(*in units of* :math:`\\mathrm{eV}`) of the particles to the base 10
		at which the spectra should be interpolated.

	Returns
	-------
	:obj:`array-like`
		Array (:code:`shape = (3,l)`) of the sampled particle spectra of
		positrons and electrons, photons, and particle not interacting
		with the IGM at the energies specified in :code:`sampling_log10E`.
	"""

	out_el = interp_log(input_log10E, input_spec_el, sampling_log10E)
	out_ph = interp_log(input_log10E, input_spec_ph, sampling_log10E)

	return np.array([out_el, out_ph])

def secondaries_from_decay(logEnergies,mass,primary):
	#from .common import sample_spectrum
	#cirelli_dir = os.path.join(data_dir, 'cirelli')
	decay_dir = '.'
	dumpername = 'decay_spectrum_of_{:s}.obj'.format(primary)

	equivalent_mass = mass
	if equivalent_mass > 5. or equivalent_mass < 0.1:
		raise ValueError('The decay spectra are only given in the range [0.1 GeV, 5 GeV] assuming DM decay. The equivalent mass for the given injection_history ({:.2g} GeV) is not in that range.'.format(equivalent_mass))

	if not hasattr(logEnergies,'__len__'):
		logEnergies = np.asarray([logEnergies])
	else:
		logEnergies = np.asarray(logEnergies)

	if not os.path.isfile( os.path.join(decay_dir, dumpername)):
		#sys.path.insert(1,cirelli_dir)
		masses, log10X, dNdLog10X_el, dNdLog10X_ph = get_decay_spectra(primary)
		total_dNdLog10X = np.asarray([dNdLog10X_el, dNdLog10X_ph])
		from interpolator import NDlogInterpolator
		#import interpolator
		interpolator = NDlogInterpolator(masses, np.rollaxis(total_dNdLog10X,1), exponent = 0, scale = 'log-log')
		dump_dict = {'dNdLog10X_interpolator':interpolator, 'log10X':log10X}
		with open(os.path.join(decay_dir, dumpername),'wb') as dump_file:
			dill.dump(dump_dict, dump_file)
	else:
		with open(os.path.join(decay_dir, dumpername),'rb') as dump_file:
			dump_dict = dill.load(dump_file)
			interpolator = dump_dict.get('dNdLog10X_interpolator')
			log10X = dump_dict.get('log10X')
	del dump_dict
	temp_log10E = log10X + np.log10(equivalent_mass)*np.ones_like(log10X)
	temp_el, temp_ph = interpolator.__call__(equivalent_mass)	
	#temp_el, temp_ph, temp_nu, temp_pr, temp_de = interpolator.__call__(equivalent_mass) / (10**temp_log10E * np.log(10))[None,:]
	ret_spectra = np.empty(shape=(2,len(logEnergies)))
	ret_spectra = sample_spectrum_decay(temp_el, temp_ph, temp_log10E, equivalent_mass, logEnergies)
	return ret_spectra

#not used anymore
def secondaries_from_simple_decay(E_secondary, E_primary, primary):
	if primary not in ['muon','pi0','piCh']:
		raise err('The "simple" decay spectrum you asked for (species: {:s}) is not (yet) known.'.format(primary))

	if not hasattr(E_secondary,'__len__'):
		E_secondary = np.asarray([E_secondary])
	else:
		E_secondary = np.asarray(E_secondary)

	#decay_dir  = os.path.join(data_dir, 'simple_decay_spectra')
	decay_dir = '.'
	dumpername = 'simple_decay_spectrum_of_{:s}.obj'.format(primary)
	original_data = '{:s}_normed.dat'.format(primary)

	if not os.path.isfile( os.path.join(decay_dir, dumpername)):
		data = np.genfromtxt( os.path.join(decay_dir, original_data), unpack = True, usecols=(0,1,2,3))
		from interpolator import NDlogInterpolator
		spec_interpolator = NDlogInterpolator(data[0,:], data[1:,:].T, exponent = 1, scale = 'lin-log')
		dump_dict = {'spec_interpolator':spec_interpolator}
		with open(os.path.join(decay_dir, dumpername),'wb') as dump_file:
			dill.dump(dump_dict, dump_file)
	else:
		with open(os.path.join(decay_dir, dumpername),'rb') as dump_file:
			dump_dict = dill.load(dump_file)
			spec_interpolator = dump_dict.get('spec_interpolator')

	x = E_secondary / E_primary
	out = spec_interpolator.__call__(x)
	#out /= (np.log(10)*E_secondary)[:,None] #keep dN/dlog10E
	return out


def PBH_mass_back_at_t(t_end, n, Mstar):
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
	def jac(mass, time, n, Mstar, scale=1):
		out = np.ones((1,1))*(-2./((n+1)*mass))*PBH_dMdt(mass*scale, time, n, Mstar)/scale # partial_(dMdt) / partial_M
		#out = np.zeros((1,1))
		return out		


	#t_start = 1.e-30
	t_start = 2.28635966e+11
	#t_end = 13.8e12*np.pi*1.e7

	scale  = 1.e12
	initial_PBH_mass = Mstar/scale

	# Workaround for dealing with **DarkOptions inside the ODE-solver.
	ODE_to_solve = lambda m,t: -1. * PBH_dMdt(m*scale, t, n, Mstar)/scale
	jac_to_use = lambda m,t: -1. * jac(m,t, n, Mstar, scale = scale)    

	#temp_t = 10**np.linspace(np.log10(time_at_z(z_start)), np.log10(time_at_z(1.)), 1e5)
	temp_t = np.logspace(np.log10(t_start), np.log10(t_end), 1e5)
	temp_mass, full_info = solve_ode(ODE_to_solve, initial_PBH_mass, temp_t, Dfun=jac_to_use, full_output=1,mxstep=10000)

	out = np.array([temp_t,temp_mass[:,0]*scale])

	return out	



def PBH_mass_at_t(initial_PBH_mass, t_end, n, Mstar):
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
	def jac(mass, time, n, Mstar, scale=1):
		out = np.ones((1,1))*(-2./((n+1)*mass))*PBH_dMdt(mass*scale, time, n, Mstar)/scale # partial_(dMdt) / partial_M
		#out = np.zeros((1,1))
		return out		


	#t_start = 1.e-30
	t_start = 2.28635966e+11
	#t_end = 13.8e12*np.pi*1.e7

	log10_ini_mass = np.log10(initial_PBH_mass)
	scale = 10**(np.floor(log10_ini_mass)+5)
	initial_PBH_mass *= 1/scale

	# Workaround for dealing with **DarkOptions inside the ODE-solver.
	ODE_to_solve = lambda m,t: PBH_dMdt(m*scale, t, n, Mstar)/scale
	jac_to_use = lambda m,t: jac(m,t, n, Mstar, scale = scale)    

	#temp_t = 10**np.linspace(np.log10(time_at_z(z_start)), np.log10(time_at_z(1.)), 1e5)
	temp_t = np.logspace(np.log10(t_start), np.log10(t_end), 1e5)
	temp_mass, full_info = solve_ode(ODE_to_solve, initial_PBH_mass, temp_t, Dfun=jac_to_use, full_output=1,mxstep=10000)

	out = np.array([temp_t,temp_mass[:,0]*scale])

	return out	

