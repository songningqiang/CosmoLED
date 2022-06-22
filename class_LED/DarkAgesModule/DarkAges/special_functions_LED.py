from __future__ import absolute_import, division, print_function

import numpy as np
import os
import sys
import dill
from scipy.special import erf
from scipy.interpolate import interp1d

from .__init__ import DarkAgesError as err
data_dir = os.path.join( os.path.dirname(os.path.realpath( __file__ )), 'data' )

def boost_factor_halos(redshift,zh,fh):
	ret = 1 + fh*erf(redshift/(1+zh))/redshift**3
	return ret

def secondaries_from_cirelli(logEnergies,mass,primary, **DarkOptions):
	from .common import sample_spectrum
	cirelli_dir = os.path.join(data_dir, 'cirelli')
	dumpername = 'cirelli_spectrum_of_{:s}.obj'.format(primary)

	injection_history = DarkOptions.get("injection_history","annihilation")
	if "decay" in injection_history:
		equivalent_mass = mass/2.
	else:
		equivalent_mass = mass
	if equivalent_mass < 5 or equivalent_mass > 1e5:
		raise err('The spectra of Cirelli are only given in the range [5 GeV, 1e2 TeV] assuming DM annihilation. The equivalent mass for the given injection_history ({:.2g} GeV) is not in that range.'.format(equivalent_mass))

	if not hasattr(logEnergies,'__len__'):
		logEnergies = np.asarray([logEnergies])
	else:
		logEnergies = np.asarray(logEnergies)

	if not os.path.isfile( os.path.join(cirelli_dir, dumpername)):
		sys.path.insert(1,cirelli_dir)
		from spectrum_from_cirelli_LED import get_cirelli_spectra
		masses, log10X, dNdLog10X_el, dNdLog10X_ph, dNdLog10X_oth = get_cirelli_spectra(primary)
		total_dNdLog10X = np.asarray([dNdLog10X_el, dNdLog10X_ph, dNdLog10X_oth])
		from .interpolator import NDlogInterpolator
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
	temp_el, temp_ph, temp_oth = interpolator.__call__(equivalent_mass) / (10**temp_log10E * np.log(10))[None,:]
	ret_spectra = np.empty(shape=(3,len(logEnergies)))
	ret_spectra = sample_spectrum(temp_el, temp_ph, temp_oth, temp_log10E, mass, logEnergies, **DarkOptions)
	return ret_spectra

def secondaries_from_simple_decay(E_secondary, E_primary, primary):
	if primary not in ['muon','pi0','piCh']:
		raise err('The "simple" decay spectrum you asked for (species: {:s}) is not (yet) known.'.format(primary))

	if not hasattr(E_secondary,'__len__'):
		E_secondary = np.asarray([E_secondary])
	else:
		E_secondary = np.asarray(E_secondary)

	decay_dir  = os.path.join(data_dir, 'simple_decay_spectra')
	dumpername = 'simple_decay_spectrum_of_{:s}.obj'.format(primary)
	original_data = '{:s}_normed.dat'.format(primary)

	if not os.path.isfile( os.path.join(decay_dir, dumpername)):
		data = np.genfromtxt( os.path.join(decay_dir, original_data), unpack = True, usecols=(0,1,2,3))
		from .interpolator import NDlogInterpolator
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
	out /= (np.log(10)*E_secondary)[:,None]
	return out

def get_decay_spectra(key):
	if not key in ['muon', 'pi0', 'piCh', 'tau']:
		raise err('Decay table only provided for muon, pi0, piCh and tau.')

	decay_dir = os.path.join(data_dir, 'decay_spectra')
	data_elec = np.genfromtxt(os.path.join(decay_dir,'secondariesfrom'+key+'.dat'), unpack=True, usecols=(0, 1, 2), skip_header=1)
	data_phot = np.genfromtxt(os.path.join(decay_dir,'secondariesfrom'+key+'.dat'), unpack=True, usecols=(0, 1, 3), skip_header=1)
	masses = np.unique(data_elec[0,:])
	log10X = np.unique(data_elec[1,:])
	dim1 = len(masses)
	dim2 = len(log10X)
	dNdlog10X_el = data_elec[2,:].reshape(dim1,dim2) #this considers only primary particle, not antiparticle
	dNdlog10X_ph = data_phot[2,:].reshape(dim1,dim2) #this consdiers both particle and antiparticle

	return masses, log10X, dNdlog10X_el, dNdlog10X_ph

def interp_log(x, y, x0):
	f = interp1d(x, y, kind = 'linear', bounds_error = False, fill_value = (0.,0.))
	return f(x0)

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

def secondaries_from_decay(logEnergies,mass,primary, **DarkOptions):
	#from .common import sample_spectrum
	decay_dir = os.path.join(data_dir, 'decay_spectra')
	dumpername = 'decay_spectrum_of_{:s}.obj'.format(primary)

	equivalent_mass = mass
	if equivalent_mass > 5. or equivalent_mass < 0.1:
		raise ValueError('The decay spectra are only given in the range [0.1 GeV, 5 GeV] assuming DM decay. The equivalent mass for the given injection_history ({:.2g} GeV) is not in that range.'.format(equivalent_mass))

	if not hasattr(logEnergies,'__len__'):
		logEnergies = np.asarray([logEnergies])
	else:
		logEnergies = np.asarray(logEnergies)

	if not os.path.isfile( os.path.join(decay_dir, dumpername)):
		masses, log10X, dNdLog10X_el, dNdLog10X_ph = get_decay_spectra(primary)
		total_dNdLog10X = np.asarray([dNdLog10X_el, dNdLog10X_ph])
		from .interpolator import NDlogInterpolator
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
	#temp_el, temp_ph = interpolator.__call__(equivalent_mass)
	#convert dN/dlgx to dN/dE	
	temp_el, temp_ph = interpolator.__call__(equivalent_mass) / (10**temp_log10E * np.log(10))[None,:]
	ret_spectra = np.empty(shape=(2,len(logEnergies)))
	ret_spectra = sample_spectrum_decay(temp_el, temp_ph, temp_log10E, equivalent_mass, logEnergies)
	return ret_spectra


def secondaries_from_cirelli_LED(logEnergies,mass,primary, **DarkOptions):

	cirelli_dir = os.path.join(data_dir, 'cirelli')
	dumpername = 'cirelli_spectrum_of_{:s}.obj'.format(primary)

	injection_history = DarkOptions.get("injection_history","annihilation")
	if "decay" in injection_history:
		equivalent_mass = mass/2.
	else:
		equivalent_mass = mass
	if equivalent_mass < 5 or equivalent_mass > 1e5:
		raise err('The spectra of Cirelli are only given in the range [5 GeV, 1e2 TeV] assuming DM annihilation. The equivalent mass for the given injection_history ({:.2g} GeV) is not in that range.'.format(equivalent_mass))

	if not hasattr(logEnergies,'__len__'):
		logEnergies = np.asarray([logEnergies])
	else:
		logEnergies = np.asarray(logEnergies)

	equivalent_mass = mass
	if equivalent_mass < 5 or equivalent_mass > 1e5:
		raise ValueError('The spectra of Cirelli are only given in the range [5 GeV, 1e2 TeV] assuming DM annihilation. The equivalent mass for the given injection_history ({:.2g} GeV) is not in that range.'.format(equivalent_mass))

	if not hasattr(logEnergies,'__len__'):
		logEnergies = np.asarray([logEnergies])
	else:
		logEnergies = np.asarray(logEnergies)

	if not os.path.isfile( os.path.join(cirelli_dir, dumpername)):
		sys.path.insert(1,cirelli_dir)
		from spectrum_from_cirelli_LED import get_cirelli_spectra
		masses, log10X, dNdLog10X_el, dNdLog10X_ph, dNdLog10X_oth = get_cirelli_spectra(primary)
		total_dNdLog10X = np.asarray([dNdLog10X_el, dNdLog10X_ph, dNdLog10X_oth])
		from .interpolator import NDlogInterpolator
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
	#temp_el, temp_ph, temp_oth = interpolator.__call__(equivalent_mass)
	#convert dN/dlgx to dN/dE	
	temp_el, temp_ph, temp_oth = interpolator.__call__(equivalent_mass) / (10**temp_log10E * np.log(10))[None,:]
	ret_spectra = np.empty(shape=(5,len(logEnergies)))
	ret_spectra = sample_spectrum_decay(temp_el, temp_ph, temp_log10E, equivalent_mass, logEnergies)
	return ret_spectra


def luminosity_accreting_bh(Energy,recipe,PBH_mass):
	if not hasattr(Energy,'__len__'):
		Energy = np.asarray([Energy])
	if recipe=='spherical_accretion':
		a = 0.5
		Ts = 0.4*511e3
		Emin = 1
		Emax = Ts
		out = np.zeros_like(Energy)
		Emin_mask = Energy > Emin
		# Emax_mask = Ts > Energy
		out[Emin_mask] = Energy[Emin_mask]**(-a)*np.exp(-Energy[Emin_mask]/Ts)
		out[~Emin_mask] = 0.
		# out[~Emax_mask] = 0.

	elif recipe=='disk_accretion':
		a = -2.5+np.log10(PBH_mass)/3.
		Emin = (10/PBH_mass)**0.5
		# print a, Emin
		Ts = 0.4*511e3
		out = np.zeros_like(Energy)
		Emin_mask = Energy > Emin
		out[Emin_mask] = Energy[Emin_mask]**(-a)*np.exp(-Energy[Emin_mask]/Ts)
		out[~Emin_mask] = 0.
		Emax_mask = Ts > Energy
		out[~Emax_mask] = 0.
	else:
		from .__init__ import DarkAgesError as err
		raise err('I cannot understand the recipe "{0}"'.format(recipe))
	# print out, Emax_mask
	return out/Energy #We will remultiply by Energy later in the code
