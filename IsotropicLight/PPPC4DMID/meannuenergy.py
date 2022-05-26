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
	dNdlog10X_el = data_elec_EW[2,:].reshape(dim1,dim2)
	dNdlog10X_ph = data_phot_EW[2,:].reshape(dim1,dim2)
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
		raise err('The spectra of Cirelli are only given in the range [5 GeV, 1e2 TeV] assuming DM annihilation. The equivalent mass for the given injection_history ({:.2g} GeV) is not in that range.'.format(equivalent_mass))

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

masses, log10X, dNdLog10X_el, dNdLog10X_ph, dNdLog10X_nu, dNdLog10X_pr, dNdLog10X_de = get_cirelli_spectra('tau')

#test normalization
#E = masses[2]
#x = 10.**log10X
#Esec = E*x
#spectot=(2*dNdLog10X_nu+2*dNdLog10X_el+dNdLog10X_ph+4*dNdLog10X_pr+2*dNdLog10X_de)
#Etot = np.trapz(spectot[2,:]*Esec, log10X)
#print(E*2)
#print(Etot)

#print(dNdLog10X_nu.shape)
#print(masses.shape)
#print(log10X.shape)
#print(masses)
#print(log10X)

#INT=np.trapz(dNdLog10X_nu, log10X, axis=1)
#print(INT)

#plt.plot(log10X, dNdLog10X_nu[0,:])
#plt.plot(log10X, dNdLog10X_nu[-1,:])
#plt.show()

#calculate average neutrino energy
x = 10.**log10X
Esec = masses[:,None] * x[None,:]
Enumean = np.trapz(Esec*dNdLog10X_nu, log10X, axis=1)
Ephmean = np.trapz(Esec*dNdLog10X_ph, log10X, axis=1)/2.
Eelmean = np.trapz(Esec*dNdLog10X_el, log10X, axis=1)
Eprmean = np.trapz(Esec*dNdLog10X_pr, log10X, axis=1)*2
Edemean = np.trapz(Esec*dNdLog10X_de, log10X, axis=1)
Etot = Enumean + Ephmean + Eelmean + Eprmean + Edemean
#print(Etot)
#plt.loglog(masses,Enumean)
#plt.loglog(masses,Ephmean)
#plt.loglog(masses,Eelmean)
#plt.loglog(masses,Eprmean)
#plt.loglog(masses,Edemean)
#plt.legend(['neutrino','photon','electron','proton','deuterium'])
#plt.ylim([1e-2,1e5])
#plt.title('light quark')
#plt.savefig('meanenergy/tau.pdf',format='pdf')
#plt.show()

dout = masses
primaries = list(column_dict.keys())
for particle in primaries:
	print(particle)
	masses, log10X, dNdLog10X_el, dNdLog10X_ph, dNdLog10X_nu, dNdLog10X_pr, dNdLog10X_de = get_cirelli_spectra(particle)
	Enumean = np.trapz(Esec*dNdLog10X_nu, log10X, axis=1)
	dout = np.vstack((dout, Enumean))

np.savetxt('meanenergy/Enumean.txt', np.transpose(dout))

primaries = list(column_dict.keys())
idx = [0,1,2,3,4,7,9,10,11,12]
primaries = [primaries[index] for index in idx]
for particle in primaries:
	masses, log10X, dNdLog10X_el, dNdLog10X_ph, dNdLog10X_nu, dNdLog10X_pr, dNdLog10X_de = get_cirelli_spectra(particle)
	Enumean = np.trapz(Esec*dNdLog10X_nu, log10X, axis=1)
	plt.loglog(masses,Enumean)
plt.ylim([1e-2,1e5])
plt.legend(primaries)
plt.xlabel('Eprimary (GeV)')
plt.ylabel('<Enu> (GeV)')
plt.savefig('meanenergy/Enumean.pdf',format='pdf')
plt.show()









