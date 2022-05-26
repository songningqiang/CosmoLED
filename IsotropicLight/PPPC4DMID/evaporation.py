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
from scipy.integrate import solve_ivp
from scipy.integrate import ode
import matplotlib.pyplot as plt

#define constants
gramtoGeV = 5.62e23
hbar = 6.582e-25 #GeV*s


def kn(n):
	#return (2.**n*math.pi**((n-3.)/2.)*math.gamma((n+3.)/2.)/(n+2.))**(1./(n+1.))
	return (8.*math.pi**(-(n+1.)/2.)*math.gamma((n+3.)/2.)/(n+2.))**(1./(n+1.))

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
		ret = ret/(2.*np.pi)/rh**2

		return -ret/gramtoGeV/hbar
	else:
		return -0.0		


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


	t_start = 1.e-30
	#t_start = 2.28635966e+11
	#t_end = 13.8e12*np.pi*1.e7

	scale  = 1.e5
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


	Parameters
	----------
	initial_PBH_mass : :obj:`float`
		Initial mass of the primordial black hole (*in units of g*)
	t_end : end time of the mass evolution, in seconds
	n : number of extra dimensions
	Mstar : Bulk Planck scale

	Returns
	-------
	:obj:`array-like`
		Array of the PBH mass at time t, the first row is time and the second row is PBH mass
	"""

	# Jacobian of the ODE for the PBH mass.
	# Needed for better performance of the ODE-solver.
	def jac(mass, time, n, Mstar, scale=1):
		out = np.ones((1,1))*(-2./((n+1)*mass))*PBH_dMdt(mass*scale, time, n, Mstar)/scale # partial_(dMdt) / partial_M
		#out = np.zeros((1,1))
		return out		


	t_start = 1.e-30
	#t_start = 2.28635966e+11
	#t_end = 13.8e12*np.pi*1.e7

	log10_ini_mass = np.log10(initial_PBH_mass)
	scale = 10**(np.floor(log10_ini_mass)+5)
	initial_PBH_mass *= 1/scale

	# Workaround for dealing with **DarkOptions inside the ODE-solver.
	ODE_to_solve = lambda m,t: PBH_dMdt(m*scale, t, n, Mstar)/scale
	jac_to_use = lambda m,t: jac(m,t, n, Mstar, scale = scale)    

	#temp_t = 10**np.linspace(np.log10(time_at_z(z_start)), np.log10(time_at_z(1.)), 1e5)
	temp_t = np.logspace(np.log10(t_start), np.log10(t_end), 100000)
	temp_mass, full_info = solve_ode(ODE_to_solve, initial_PBH_mass, temp_t, Dfun=jac_to_use, full_output=1,mxstep=10000)

	out = np.array([temp_t,temp_mass[:,0]*scale])

	return out	


def PBH_mass_at_t_detail(initial_PBH_mass, t_end, n, Mstar):
	u"""Solves the ODE for the PBH mass (:meth:`PBH_dMdt <DarkAges.evaporator.PBH_dMdt>`)
	and returns the masses at the redshifts given by the input :code:`redshift`


	Parameters
	----------
	initial_PBH_mass : :obj:`float`
		Initial mass of the primordial black hole (*in units of g*)
	t_end : end time of the mass evolution, in seconds
	n : number of extra dimensions
	Mstar : Bulk Planck scale

	Returns
	-------
	:obj:`array-like`
		Array of the PBH mass at time t, the first row is time and the second row is PBH mass
	"""


	# Jacobian of the ODE for the PBH mass.
	# Needed for better performance of the ODE-solver.
	def jac(mass, time, n, Mstar, scale=1):
		out = np.ones((1,1))*(-2./((n+1)*mass))*PBH_dMdt(mass*scale, time, n, Mstar)/scale # partial_(dMdt) / partial_M
		#out = np.zeros((1,1))
		return out		


	t_start = 1.e-30
	#t_start = 2.28635966e+11
	#t_end = 13.8e12*np.pi*1.e7

	log10_ini_mass = np.log10(initial_PBH_mass)
	scale = 10**(np.floor(log10_ini_mass)+5)
	initial_PBH_mass *= 1/scale

	# Workaround for dealing with **DarkOptions inside the ODE-solver.
	ODE_to_solve = lambda t,m: PBH_dMdt(m*scale, t, n, Mstar)/scale
	jac_to_use = lambda t,m: jac(m,t, n, Mstar, scale = scale)    

	"""
	#temp_t = 10**np.linspace(np.log10(time_at_z(z_start)), np.log10(time_at_z(1.)), 1e5)
	temp_t = np.logspace(np.log10(t_start), np.log10(t_end), 100000)
	#solver = ode(ODE_to_solve, jac = jac_to_use).set_integrator('lsoda', nsteps = 1, rtol = 1.e-2, max_order_s = 2, max_order_ns = 4)
	solver = ode(ODE_to_solve, jac = jac_to_use).set_integrator('dopri5', nsteps = 1, rtol = 1.e-2)
	solver.set_initial_value(initial_PBH_mass, t_start)
	idx = 1
	sol = []
	#while solver.successful() and idx < 100000:
	#	solver.integrate(temp_t[idx], step = True)
	#	sol.append([solver.t, solver.y[0]])
	#	idx = idx + 1
	while solver.successful() and solver.t < t_end:
		solver.integrate(t_end, step = True)
		sol.append([solver.t, solver.y[0]])

	sol = np.array(sol)
	sol[:,1] = sol[:,1]*scale
	"""
	#LSODA takes forever
	sol = solve_ivp(ODE_to_solve, [t_start, t_end], [initial_PBH_mass], jac = jac_to_use, method = 'RK45', dense_output=True, rtol = 1.e-4)
	#sol = solve_ivp(ODE_to_solve, [t_start, t_end], [initial_PBH_mass], method = 'RK45', dense_output=True, rtol = 1e-2)
	sol = np.array([sol.t, sol.y[0]*scale])

	return sol.T

def PBH_mass_at_t_ode(initial_PBH_mass, t_end, n, Mstar):
	u"""Solves the ODE for the PBH mass (:meth:`PBH_dMdt <DarkAges.evaporator.PBH_dMdt>`)
	and returns the masses at the redshifts given by the input :code:`redshift`


	Parameters
	----------
	initial_PBH_mass : :obj:`float`
		Initial mass of the primordial black hole (*in units of g*)
	t_end : end time of the mass evolution, in seconds
	n : number of extra dimensions
	Mstar : Bulk Planck scale

	Returns
	-------
	:obj:`array-like`
		Array of the PBH mass at time t, the first row is time and the second row is PBH mass
	"""


	# Jacobian of the ODE for the PBH mass.
	# Needed for better performance of the ODE-solver.
	def jac(mass, time, n, Mstar, scale=1):
		out = np.ones((1,1))*(-2./((n+1)*mass))*PBH_dMdt(mass*scale, time, n, Mstar)/scale # partial_(dMdt) / partial_M
		#out = np.zeros((1,1))
		return out		


	t_start = 0.
	#t_start = 2.28635966e+11
	#t_end = 13.8e12*np.pi*1.e7

	log10_ini_mass = np.log10(initial_PBH_mass)
	scale = 10**(np.floor(log10_ini_mass)+5)
	#scale = 1
	initial_PBH_mass *= 1/scale

	ODE_to_solve = lambda t,m: PBH_dMdt(m*scale, t, n, Mstar)/scale
	jac_to_use = lambda t,m: jac(m,t, n, Mstar, scale = scale)    

	#vode turns out to be the best
	solver = ode(ODE_to_solve, jac = jac_to_use).set_integrator('vode', nsteps = 1, first_step = 1.e-12, method = "Adams", rtol = 1.e-12, order = 5)
	#solver = ode(ODE_to_solve, jac = jac_to_use).set_integrator('lsoda', nsteps = 1, rtol = 1.e-4, max_order_ns = 4, max_order_s = 2)
	solver.set_initial_value(initial_PBH_mass, t_start)
	sol = []
	sol.append([t_start, initial_PBH_mass*scale])
	while solver.successful() and solver.t < t_end:
		solver.integrate(t_end, step = True)
		sol.append([solver.t, solver.y[0]*scale])

	sol = np.array(sol)

	return sol	


#find the average quark energy, assuming quark is massless
def dNdM(PBH_mass, n, Mstar):
	u"""find dN/dM ~ 1/T_H/sum_i xi_i(M)
	PBH_mass: BH mass in grams
	n: number of exra dimensions
	Mstar: Bulk Planck scale in GeV
	Retures: dN/dM in 1/GeV
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

		#add gravitons, this is sum_i xi_i (dimensionless)
		ret += xi_graviton[n-1]

		return -1/ret/TH
	else:
		return -0.


#this is (int dE/dt / int dN/dt) / T_H, assuming a massless fermion
fEqmean = [2.7766, 2.7311, 2.7512, 2.7935, 2.8399, 2.8832]
fhadronmean = [2.1812, 2.1065, 2.0883, 2.0986, 2.1216, 2.1495]

def Eqmean(initial_PBH_mass, n, Mstar):
	u"""find <E_q> in Eq. 10 of 2006.03608
	initial_PBH_mass: initial BH mass in grams M_i
	n: number of exra dimensions
	Mstar: Bulk Planck scale in GeV
	Retures: <E_q>/T_i
	"""

	#if TH < QCD scale, no relativistic quark emission
	Lam_QCD = 0.3
	#find the threshold mass where TH = QCD scale
	Mth = get_mass_from_temperature(Lam_QCD,n,Mstar)
	#print(Mth)

	#Mf > 0 to avoid divergence at 0
	Mf = Mstar/gramtoGeV
	if initial_PBH_mass > Mth:
		numerator = (quad(lambda x: np.heaviside(get_temperature_from_mass(x,n,Mstar)-Lam_QCD, 1)*get_temperature_from_mass(x,n,Mstar)*dNdM(x, n, Mstar), Mth, Mf))[0]
	else:
		numerator = (quad(lambda x: np.heaviside(get_temperature_from_mass(x,n,Mstar)-Lam_QCD, 1)*get_temperature_from_mass(x,n,Mstar)*dNdM(x, n, Mstar), initial_PBH_mass, Mf))[0]
	if initial_PBH_mass > Mth:
		denominator = (quad(lambda x: np.heaviside(get_temperature_from_mass(x,n,Mstar)-Lam_QCD, 1)*dNdM(x, n, Mstar), Mth, Mf))[0]
	else:
		denominator = (quad(lambda x: np.heaviside(get_temperature_from_mass(x,n,Mstar)-Lam_QCD, 1)*dNdM(x, n, Mstar), initial_PBH_mass, Mf))[0]
	Ti = get_temperature_from_mass(initial_PBH_mass,n,Mstar)
	return numerator/denominator/Ti



#generate output

n = 6 #number of extra dimensions
Mstar = 1e4 #bulk Planck scale

"""
#find BBN factor
Lam_QCD = 0.3
Mth = get_mass_from_temperature(Lam_QCD,n,Mstar)
print("{:e}".format(Mth))

Ms = np.logspace(np.log10(Mth*1e-5), np.log10(Mth*2), 1000)
Eqs = np.asarray(np.vectorize(Eqmean, excluded = ['n','Mstar']).__call__(Ms, n, Mstar))
print(Eqs[0]*fEqmean[n-1])
print(Eqs[0]*fhadronmean[n-1])
plt.semilogx(Ms, Eqs*fEqmean[n-1])
plt.semilogx(Ms, Eqs*fhadronmean[n-1])
plt.legend(['quark mean n = %d' %n, 'hadron mean n = %d' %n])
plt.xlabel('$M_i$[g]')
plt.ylabel('$<E_q>/T_i$')
#plt.savefig('Eqmean_n%d.pdf'%n, format = 'pdf')
plt.show()
sys.exit(0)
"""


#MBH = 1.e3 #intial black hole mass in grams
#t_end = 13.8e9*np.pi*1.e7 #eolve the PBH mass unitil time t (in s)
t_end = 13.8e9*np.pi*1.e50 #eolve the PBH mass unitil time t (in s)
#MBHoft = PBH_mass_back_at_t(t_end, n, Mstar)
#plt.loglog(MBHoft[0,:], MBHoft[1,:],'b')
#MBH = MBHoft[1,-1]
#print(MBH)
#MBH = 2.5e7


#MBHoft = PBH_mass_at_t(MBH, t_end, n, Mstar)
#np.savetxt('testmass.txt', MBHoft.T)
#plt.loglog(MBHoft[0,:], MBHoft[1,:],'r')

#MBHoft = PBH_mass_at_t_detail(MBH, t_end, n, Mstar)
#MBHoft = PBH_mass_at_t_ode(MBH, t_end, n, Mstar)
#print(MBHoft)
#plt.loglog(MBHoft[:,0], MBHoft[:,1],'r')
#plt.xlim(2,t_end)
#plt.xticks([1e2, 1e5, 1e8, 1e11, 1e14, 1e17])
#np.savetxt('testmass.txt', MBHoft)
#plt.show()


Ms = np.logspace(-5, np.log10(1.e26), 100)
t_sol = []
for M in Ms:
	MBHoft = PBH_mass_at_t_ode(M, t_end, n, Mstar)
	ts = MBHoft[:,0]
	ts = ts[MBHoft[:,1]>0]
	t_sol.append([M, ts[-1]])
t_sol = np.array(t_sol)
np.savetxt('BHlifetime_n%d.txt'%n, t_sol)
plt.loglog(t_sol[:,0], t_sol[:,1])
plt.show()

