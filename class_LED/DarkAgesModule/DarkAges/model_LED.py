u"""
.. module:: model
   :synopsis: Definition of the model-class and its derived classes for annihilation, decay, accretion and evaporation of primordial black holes
.. moduleauthor:: Patrick Stoecker <stoecker@physik.rwth-aachen.de>

Contains the definition of the base model class :class:`model <DarkAges.model.model>`,
with the basic functions

* :func:`calc_f` to calculate :math:`f(z)`, given an instance of
  :class:`transfer <DarkAges.transfer.transfer>` and
* :func:`model.save_f` to run :func:`calc_f` and saved it in a file.

Also contains derived classes

* :class:`annihilating_model <DarkAges.model.annihilating_model>`
* :class:`annihilating_halos_model <DarkAges.model.annihilating_halos_model>`
* :class:`decaying_model <DarkAges.model.decaying_model>`
* :class:`evaporating_model <DarkAges.model.evaporating_model>`
* :class:`accreting_model <DarkAges.model.accreting_model>`

for the most common energy injection histories.

"""

from __future__ import absolute_import, division, print_function
from builtins import range, object

from .transfer import transfer
from .common_LED import f_function
from .__init__ import DarkAgesError, get_logEnergies, get_redshift, print_info
import numpy as np
import sys
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt

gramtoGeV = 5.62e23
hbar = 6.582e-25 #GeV*s

class model(object):
	u"""
	Base class to calculate :math:`f(z)` given the injected spectrum
	:math:`\mathrm{d}N / \mathrm{d}E` as a function of *kinetic energy* :math:`E`
	and *redshift* :math:`z+1`
	"""

	def __init__(self, spec_electrons, spec_photons, normalization, logEnergies, alpha=3):
		u"""
		Parameters
		----------
		spec_electrons : :obj:`array-like`
			Array of shape (m,n) containing :math:`\mathrm{d}N / \mathrm{d}E` of
			**electrons** at given redshift :math:`z+1` and
			kinetic energy :math:`E`
		spec_photons : :obj:`array-like`
			Array of shape (m,n) containing :math:`\mathrm{d}N / \mathrm{d}E` of
			**photons** at given redshift :math:`z+1` and
			kinetic energy :math:`E`
		normalization : :obj:`array-like`
			Array of shape (m) with the normalization of the given spectra
			at each given :math:`z_\mathrm{dep.}`.
			(e.g constant array with entries :math:`2m_\mathrm{DM}` for DM-annihilation
			or constant array with entries :math:`m_\mathrm{DM}` for decaying DM)
		alpha : :obj:`int`, :obj:`float`, *optional*
			Exponent to specify the comoving scaling of the
			injected spectra.
			(3 for annihilation and 0 for decaying species
			`c.f. ArXiv1801.01871 <https://arxiv.org/abs/1801.01871>`_).
			If not specified annihilation is assumed.
		"""

		self.logEnergies = logEnergies
		self.spec_electrons = spec_electrons
		self.spec_photons = spec_photons
		self.normalization = normalization
		self.alpha_to_use = alpha

	def calc_f(self, transfer_instance, **DarkOptions):
		u"""Returns :math:`f(z)` for a given set of transfer functions
		:math:`T(z_{dep}, E, z_{inj})`

		Parameters
		----------
		transfer_instance : :obj:`class`
			Initialized instace of :class:`transfer <DarkAges.transfer.transfer>`

		Returns
		-------
		:obj:`array-like`
			Array (:code:`shape=(2,n)`) containing :math:`z_\mathrm{dep}+1` in the first column
			and :math:`f(z_\mathrm{dep})` in the second column.
		"""

		if not isinstance(transfer_instance, transfer):
			raise DarkAgesError('You did not include a proper instance of the class "transfer"')
		else:
			red = transfer_instance.z_deposited

			f_func = f_function(transfer_instance.log10E,self.logEnergies, transfer_instance.z_injected,
                                transfer_instance.z_deposited, self.normalization,
                                transfer_instance.transfer_phot,
                                transfer_instance.transfer_elec,
                                self.spec_photons, self.spec_electrons, alpha=self.alpha_to_use, **DarkOptions)

			return np.array([red, f_func], dtype=np.float64)

	def save_f(self,transfer_instance, filename, **DarkOptions):
		u"""Saves the table :math:`z_\mathrm{dep.}`, :math:`f(z_\mathrm{dep})` for
		a given set of transfer functions :math:`T(z_{dep}, E, z_{inj})` in a file.

		Parameters
		----------
		transfer_instance : :obj:`class`
			Initialized instace of :class:`transfer <DarkAges.transfer.transfer>`
		filename : :obj:`str`
			Self-explanatory
		"""

		f_function = self.calc_f(transfer_instance,**DarkOptions)
		file_out = open(filename, 'w')
		file_out.write('#z_dep\tf(z)')
		for i in range(len(f_function[0])):
			file_out.write('\n{:.2e}\t{:.4e}'.format(f_function[0,i],f_function[1,i]))
		file_out.close()
		print_info('Saved effective f(z)-curve under "{0}"'.format(filename))

class annihilating_model(model):
	u"""Derived instance of the class :class:`model <DarkAges.model.model>` for the case of an annihilating
	species.

	Inherits all methods of :class:`model <DarkAges.model.model>`
	"""

	def __init__(self,ref_el_spec,ref_ph_spec,ref_oth_spec,m,logEnergies = None,redshift=None, **DarkOptions):
		u"""
		At initialization the reference spectra are read and the double-differential
		spectrum :math:`\\frac{\\mathrm{d}^2 N(t,E)}{\\mathrm{d}E\\mathrm{d}t}` needed for
		the initialization inherited from :class:`model <DarkAges.model.model>` is calculated by

		.. math::
			\\frac{\\mathrm{d}^2 N(t,E)}{\\mathrm{d}E\\mathrm{d}t} = C \\cdot\\frac{\\mathrm{d}N(E)}{\\mathrm{d}E}

		where :math:`C` is a constant independent of :math:`t` (:math:`z`) and :math:`E`

		Parameters
		----------
		ref_el_spec : :obj:`array-like`
			Reference spectrum (:code:`shape = (k,l)`) :math:`\mathrm{d}N / \mathrm{d}E` of **electrons**
		ref_ph_spec : :obj:`array-like`
			Reference spectrum (:code:`shape = (k,l)`) :math:`\mathrm{d}N / \mathrm{d}E` of **photons**
		ref_oth_spec : :obj:`array-like`
			Reference spectrum (:code:`shape = (k,l)`) :math:`\mathrm{d}N / \mathrm{d}E` of particles
			not interacting with the erly IGM (e.g. **protons** and **neutrinos**).
			This is neede for the proper normalization of the electron- and photon-spectra.
		m : :obj:`float`
			Mass of the DM-candidate (*in units of* :math:`\\mathrm{GeV}`)
		logEnergies : :obj:`array-like`, optional
			Array (:code:`shape = (l)`) of the logarithms of the kinetic energies of the particles
			(*in units of* :math:`\\mathrm{eV}`) to the base 10.
			If not specified, the standard array provided by
			:class:`the initializer <DarkAges.__init__>`  is taken.
		redshift : :obj:`array-like`, optional
			Array (:code:`shape = (k)`) with the values of :math:`z+1`. Used for
			the calculation of the double-differential spectra.
			If not specified, the standard array provided by
			:mod:`the initializer <DarkAges.__init__>`  is taken.
		"""

		if logEnergies is None:
			logEnergies = get_logEnergies()
		if redshift is None:
			redshift = get_redshift()

		tot_spec = ref_el_spec + ref_ph_spec + ref_oth_spec

		norm_by = DarkOptions.get('normalize_spectrum_by','energy_integral')
		if norm_by == 'energy_integral':
			from .common import trapz, logConversion
			E = logConversion(logEnergies)
			if len(E) > 1:
				normalization = trapz(tot_spec*E**2*np.log(10), logEnergies)*np.ones_like(redshift)
			else:
				normalization = (tot_spec*E)[0]
		elif norm_by == 'mass':
			normalization = np.ones_like(redshift)*(2*m)
		else:
			raise DarkAgesError('I did not understand your input of "normalize_spectrum_by" ( = {:s}). Please choose either "mass" or "energy_integral"'.format(norm_by))

		spec_electrons = np.zeros((len(tot_spec),len(redshift)))
		spec_photons = np.zeros((len(tot_spec),len(redshift)))
		spec_electrons[:,:] = ref_el_spec[:,None]
		spec_photons[:,:] = ref_ph_spec[:,None]

		model.__init__(self, spec_electrons, spec_photons, normalization,logEnergies, 3)

class annihilating_halos_model(model):
	def __init__(self,ref_el_spec,ref_ph_spec,ref_oth_spec,m,zh,fh,logEnergies=None,redshift=None, **DarkOptions):

		from .special_functions import boost_factor_halos

		def scaling_boost_factor(redshift,spec_point,zh,fh):
			ret = spec_point*boost_factor_halos(redshift,zh,fh)
			return ret

		if logEnergies is None:
			logEnergies = get_logEnergies()
		if redshift is None:
			redshift = get_redshift()

		tot_spec = ref_el_spec + ref_ph_spec + ref_oth_spec

		norm_by = DarkOptions.get('normalize_spectrum_by','energy_integral')
		if norm_by == 'energy_integral':
			from .common import trapz, logConversion
			E = logConversion(logEnergies)
			if len(E) > 1:
				normalization = trapz(tot_spec*E**2*np.log(10), logEnergies)*np.ones_like(redshift)
			else:
				normalization = (tot_spec*E)[0]
		elif norm_by == 'mass':
			normalization = np.ones_like(redshift)*(2*m)
		else:
			raise DarkAgesError('I did not understand your input of "normalize_spectrum_by" ( = {:s}). Please choose either "mass" or "energy_integral"'.format(norm_by))
		normalization /= boost_factor_halos(redshift,zh,fh)

		spec_electrons = np.vectorize(scaling_boost_factor).__call__(redshift[None,:],ref_el_spec[:,None],zh,fh)
		spec_photons = np.vectorize(scaling_boost_factor).__call__(redshift[None,:],ref_ph_spec[:,None],zh,fh)

		model.__init__(self, spec_electrons, spec_photons, normalization, logEnergies,3)


class decaying_model(model):
	u"""Derived instance of the class :class:`model <DarkAges.model.model>` for the case of a decaying
	species.

	Inherits all methods of :class:`model <DarkAges.model.model>`
	"""

	def __init__(self,ref_el_spec,ref_ph_spec,ref_oth_spec,m,t_dec,logEnergies=None,redshift=None, **DarkOptions):
		u"""At initialization the reference spectra are read and the double-differential
		spectrum :math:`\\frac{\\mathrm{d}^2 N(t,E)}{\\mathrm{d}E\\mathrm{d}t}` needed for
		the initialization inherited from :class:`model <DarkAges.model.model>` is calculated by

		.. math::
			\\frac{\\mathrm{d}^2 N(t,E)}{\\mathrm{d}E\\mathrm{d}t} = C \\cdot\\exp{\\left(\\frac{-t(z)}{\\tau}\\right)} \\cdot \\frac{\\mathrm{d}N(E)}{\\mathrm{d}E}

		where :math:`C` is a constant independent of :math:`t` (:math:`z`) and :math:`E`

		Parameters
		----------
		ref_el_spec : :obj:`array-like`
			Reference spectrum (:code:`shape = (k,l)`) :math:`\mathrm{d}N / \mathrm{d}E` of **electrons**
		ref_ph_spec : :obj:`array-like`
			Reference spectrum (:code:`shape = (k,l)`) :math:`\mathrm{d}N / \mathrm{d}E` of **photons**
		ref_oth_spec : :obj:`array-like`
			Reference spectrum (:code:`shape = (k,l)`) :math:`\mathrm{d}N / \mathrm{d}E` of particles
			not interacting with the early IGM (e.g. **protons** and **neutrinos**).
			This is needed for the proper normalization of the electron- and photon-spectra.
		m : :obj:`float`
			Mass of the DM-candidate (*in units of* :math:`\\mathrm{GeV}`)
		t_dec : :obj:`float`
			Lifetime (Time after which the number of particles dropped down to
			a factor of :math:`1/e`) of the DM-candidate
		logEnergies : :obj:`array-like`, optional
			Array (:code:`shape = (l)`) of the logarithms of the kinetic energies of the particles
			(*in units of* :math:`\\mathrm{eV}`) to the base 10.
			If not specified, the standard array provided by
			:class:`the initializer <DarkAges.__init__>` is taken.
		redshift : :obj:`array-like`, optional
			Array (:code:`shape = (k)`) with the values of :math:`z+1`. Used for
			the calculation of the double-differential spectra.
			If not specified, the standard array provided by
			:class:`the initializer <DarkAges.__init__>` is taken.
		"""

		def _decay_scaling(redshift, spec_point, lifetime):
			from .common import time_at_z
			ret = spec_point*np.exp(-time_at_z(redshift) / lifetime)
			return ret

		if logEnergies is None:
			logEnergies = get_logEnergies()
		if redshift is None:
			redshift = get_redshift()

		tot_spec = ref_el_spec + ref_ph_spec + ref_oth_spec

		norm_by = DarkOptions.get('normalize_spectrum_by','energy_integral')
		if norm_by == 'energy_integral':
			from .common import trapz, logConversion
			E = logConversion(logEnergies)
			if len(E) > 1:
				normalization = trapz(tot_spec*E**2*np.log(10), logEnergies)*np.ones_like(redshift)
			else:
				normalization = (tot_spec*E)[0]
		elif norm_by == 'mass':
			normalization = np.ones_like(redshift)*(m)
		else:
			raise DarkAgesError('I did not understand your input of "normalize_spectrum_by" ( = {:s}). Please choose either "mass" or "energy_integral"'.format(norm_by))

		spec_electrons = np.vectorize(_decay_scaling).__call__(redshift[None,:], ref_el_spec[:,None], t_dec)
		spec_photons = np.vectorize(_decay_scaling).__call__(redshift[None,:], ref_ph_spec[:,None], t_dec)

		model.__init__(self, spec_electrons, spec_photons, normalization, logEnergies,0)

class evaporating_model(model):
	u"""Derived instance of the class :class:`model <DarkAges.model.model>` for the case of evaporating
	primordial black holes (PBH) as a candidate of DM

	Inherits all methods of :class:`model <DarkAges.model.model>`
	"""

	def __init__(self, PBH_mass_ini, logEnergies=None, redshift=None, n=6, Mstar=1e4, **DarkOptions):
		u"""
		At initialization evolution of the PBH mass is calculated with
		:func:`PBH_mass_at_z <DarkAges.evaporator.PBH_mass_at_z>` and the
		double-differential spectrum :math:`\mathrm{d}^2 N(z,E) / \mathrm{d}E\mathrm{d}z`
		needed for the initialization inherited from :class:`model <DarkAges.model.model>` is calculated
		according to :func:`PBH_spectrum <DarkAges.evaporator.PBH_spectrum>`

		Parameters
		----------
		PBH_mass_ini : :obj:`float`
			Initial mass of the primordial black hole (*in units of* :math:`\\mathrm{g}`)
		logEnergies : :obj:`array-like`, optional
			Array (:code:`shape = (l)`) of the logarithms of the kinetic energies of the particles
			(*in units of* :math:`\\mathrm{eV}`) to the base 10.
			If not specified, the standard array provided by
			:class:`the initializer <DarkAges.__init__>` is taken.
		redshift : :obj:`array-like`, optional
			Array (:code:`shape = (k)`) with the values of :math:`z+1`. Used for
			the calculation of the double-differential spectra.
			If not specified, the standard array provided by
			:class:`the initializer <DarkAges.__init__>` is taken.
		"""

		if Mstar == 'Mpl' or Mstar == 'mpl':
			Mstar = 1.22e19
		else:
			Mstar = float(Mstar)

		from .evaporator_LED import PBH_spectrum_at_m, PBH_primary_spectrum, PBH_mass_at_z, PBH_dMdt
		from .common_LED import trapz, logConversion, time_at_z, nan_clean


		include_secondaries=DarkOptions.get('PBH_with_secondaries',True)

		if logEnergies is None:
			logEnergies = get_logEnergies()
		if redshift is None:
			redshift = get_redshift()

		mass_at_z = PBH_mass_at_z(PBH_mass_ini, n, Mstar, redshift=redshift, **DarkOptions)
		dMdt_at_z = (-1)*np.vectorize(PBH_dMdt).__call__(mass_at_z[-1,:],np.ones_like(mass_at_z[0,:]),n,Mstar,scale=1,**DarkOptions)

		E = logConversion(logEnergies)
		E_sec = 1e-9*E
		E_prim = 1e-9*E

		normalization = dMdt_at_z
		M_BH = mass_at_z[-1,:]
		

		Ngrid = 50 #number of energy bins
		Eprimary = np.logspace(np.log10(5.),5.,Ngrid) #Cirelli table only allows particle energy between 5GeV and 100TeV
		logEprimary = np.log10(Eprimary)

		Edecay = np.logspace(np.log10(0.1),np.log10(5.),Ngrid) #Decay table deals with particle energy between 0.1GeV and 5eV
		Edecay[-1] = 5. #to avoid overflow
		logEdecay = np.log10(Edecay)	

		#*2 to include positrons
		prim_spec_el = PBH_spectrum_at_m( mass_at_z[-1,:], logEnergies, n, Mstar, 'electron', **DarkOptions) * 2
		prim_spec_ph = PBH_spectrum_at_m( mass_at_z[-1,:], logEnergies, n, Mstar, 'photon', **DarkOptions)

		prim_ph = prim_spec_ph[:,-1]/gramtoGeV/hbar/1e18
		prim_el = prim_spec_el[:,-1]/gramtoGeV/hbar/1e18


		# full spectra (including secondaries)
		# the secondary electrons from Cirelli has already been multiplied by 2
		me = 5.110e-4
		el_from_primary = np.zeros((len(E_sec),len(M_BH)))
		ph_from_primary = np.zeros((len(E_sec),len(M_BH)))		
		if include_secondaries:
			from .special_functions_LED import secondaries_from_cirelli_LED,  secondaries_from_decay
			prim_spec_el[E_prim+me>5,:] = 0.
			prim_spec_ph[E_prim>5,:] = 0.
			primary_list = ['electron', 'muon', 'tau', 'charm', 'bottom', 'top', 'gluon', 'wboson', 'zboson', 'photon', 'higgs', 'nue', 'numu', 'nutau']
			for ptype in primary_list:
				if ptype in ['nue', 'numu', 'nutau']:
					spec = np.asarray(np.vectorize(PBH_primary_spectrum, excluded = ['ptype','n','Mstar']).__call__(Eprimary[:,None], M_BH[None,:], n, Mstar, 'neutrino'))
				else:
					spec = np.asarray(np.vectorize(PBH_primary_spectrum, excluded = ['ptype','n','Mstar']).__call__(Eprimary[:,None], M_BH[None,:], n, Mstar, ptype))
				spec = spec*np.log(10)*Eprimary[:,None]
				temp_el = np.zeros((len(Eprimary),len(E_sec)))
				temp_ph = np.zeros((len(Eprimary),len(E_sec)))
				for idx, E in enumerate(Eprimary):
					ret_spectra = secondaries_from_cirelli_LED(logEnergies-9,E,ptype) #-9 to convert to GeVe
					temp_el[idx,:] = ret_spectra[0,:]
					temp_ph[idx,:] = ret_spectra[1,:]
				el_from_primary += np.trapz((temp_el.T)[:,:,None]*spec[None,:,:], logEprimary, axis = 1)
				ph_from_primary += np.trapz((temp_ph.T)[:,:,None]*spec[None,:,:], logEprimary, axis = 1)


			#light quarks
			spec = np.asarray(np.vectorize(PBH_primary_spectrum, excluded = ['ptype','n','Mstar']).__call__(Eprimary[:,None], M_BH[None,:], n, Mstar, 'up')) + \
				np.asarray(np.vectorize(PBH_primary_spectrum, excluded = ['ptype','n','Mstar']).__call__(Eprimary[:,None], M_BH[None,:], n, Mstar, 'down')) + \
				np.asarray(np.vectorize(PBH_primary_spectrum, excluded = ['ptype','n','Mstar']).__call__(Eprimary[:,None], M_BH[None,:], n, Mstar, 'strange'))
			spec = spec*np.log(10)*Eprimary[:,None]
			temp_el = np.zeros((len(Eprimary),len(E_sec)))
			temp_ph = np.zeros((len(Eprimary),len(E_sec)))
			for idx, E in enumerate(Eprimary):
				ret_spectra = secondaries_from_cirelli_LED(logEnergies-9,E,'quark')
				temp_el[idx,:] = ret_spectra[0,:]
				temp_ph[idx,:] = ret_spectra[1,:]
			el_from_primary += np.trapz((temp_el.T)[:,:,None]*spec[None,:,:], logEprimary, axis = 1)
			ph_from_primary += np.trapz((temp_ph.T)[:,:,None]*spec[None,:,:], logEprimary, axis = 1)
			
			#pions, muon and tau decay below 5 GeV
			primary_list = ['muon', 'pi0', 'piCh', 'tau']
			for ptype in primary_list:
				spec = np.asarray(np.vectorize(PBH_primary_spectrum, excluded = ['ptype','n','Mstar']).__call__(Edecay[:,None], M_BH[None,:], n, Mstar, ptype))
				spec = spec*np.log(10)*Edecay[:,None]
				temp_el = np.zeros((len(Edecay),len(E_sec)))
				temp_ph = np.zeros((len(Edecay),len(E_sec)))
				for idx, E in enumerate(Edecay):
					ret_spectra = secondaries_from_decay(logEnergies-9,E,ptype)
					temp_el[idx,:] = ret_spectra[0,:]
					temp_ph[idx,:] = ret_spectra[1,:]
				#add positrons for low energy decay
				el_from_primary += 2*np.trapz((temp_el.T)[:,:,None]*spec[None,:,:], logEdecay, axis = 1)
				if ptype == 'pi0' or ptype == 'tau':		
					ph_from_primary += np.trapz((temp_ph.T)[:,:,None]*spec[None,:,:], logEdecay, axis = 1)
				else: #for decay spectra consider the contribution to photons from both particles and antiparticles
					ph_from_primary += 2*np.trapz((temp_ph.T)[:,:,None]*spec[None,:,:], logEdecay, axis = 1)
				
		spec_el = prim_spec_el
		spec_ph = prim_spec_ph
		if include_secondaries:
			spec_el += el_from_primary
			spec_ph += ph_from_primary

		spec_el =  nan_clean(spec_el)
		spec_ph = nan_clean(spec_ph)

		#convert to the unit at ExoClass, 1e18 factor comes from E^2 integral using eV instead of GeV
		spec_el = spec_el/gramtoGeV/hbar/1e18
		spec_ph = spec_ph/gramtoGeV/hbar/1e18
		

		model.__init__(self, spec_el, spec_ph, normalization, logEnergies,0)

class accreting_model(model):
	u"""Derived instance of the class :class:`model <DarkAges.model.model>` for
	the case of accreting primordial black holes (PBH) as a candidate of DM.

	Inherits all methods of :class:`model <DarkAges.model.model>`
	"""

	def __init__(self, PBH_mass, recipe, logEnergies=None, redshift=None, **DarkOptions):
		u"""At initialization the reference spectra are read and the luminosity
		spectrum :math:`L_{\\omega}` needed for the initialization inherited
		from :class:`model <DarkAges.model.model>` is calculated by

		.. math::
			L_{\\omega} = \\Theta(\\omega -\\omega_\\mathrm{min})w^{-a}\\exp(-\\frac{\\omega}{T_s})

		where :math:`T_s\\simeq 200\\,\\mathrm{keV}`, :math:`a=-2.5+\\frac{\\log(M)}{3}` and
		:math:`\\omega_\\mathrm{min} = \\left(\\frac{10}{M}\\right)^{\\frac{1}{2}}`
		if :code:`recipe = disk_accretion` or

		..  math::
			L_\\omega = w^{-a}\\exp(-\\frac{\\omega}{T_s})

		where :math:`T_s\\simeq 200\\,\\mathrm{keV}` if
		:code:`recipe = spherical_accretion`.

		Parameters
		----------
		PBH_mass : :obj:`float`
			Mass of the primordial black hole (*in units of* :math:`M_\\odot`)
		recipe : :obj:`string`
			Recipe setting the luminosity and the rate of the accretion
			(`spherical_accretion` taken from 1612.05644 and `disk_accretion`
			from 1707.04206)
		logEnergies : :obj:`array-like`, optional
			Array (:code:`shape = (l)`) of the logarithms of the kinetic energies of the particles
			(*in units of* :math:`\\mathrm{eV}`) to the base 10.
			If not specified, the standard array provided by
			:class:`the initializer <DarkAges.__init__>` is taken.
		redshift : :obj:`array-like`, optional
			Array (:code:`shape = (k)`) with the values of :math:`z+1`. Used for
			the calculation of the double-differential spectra.
			If not specified, the standard array provided by
			:class:`the initializer <DarkAges.__init__>` is taken.
		"""

		if logEnergies is None:
			logEnergies = get_logEnergies()
		if redshift is None:
			redshift = get_redshift()

		from .common import trapz,  logConversion
		from .special_functions import luminosity_accreting_bh
		E = logConversion(logEnergies)
		spec_ph = luminosity_accreting_bh(E,recipe,PBH_mass)
		spec_el = np.zeros_like(spec_ph)
		spec_oth = np.zeros_like(spec_ph)
		normalization = trapz((spec_ph+spec_el)*E**2*np.log(10),logEnergies)*np.ones_like(redshift)

		spec_photons = np.zeros((len(spec_el),len(redshift)))
		spec_photons[:,:] = spec_ph[:,None]
		spec_electrons = np.zeros((len(spec_el),len(redshift)))

		model.__init__(self, spec_electrons, spec_photons, normalization, logEnergies, 0)
