#produce the secondaries of muon and pions from hazma, in hazma the energy is in MeV

import numpy as np
from hazma.decay import neutral_pion as dnde_pi0
from hazma.decay import muon as dnde_mu
from hazma.decay import charged_pion as dnde_pi
from scipy import integrate
import matplotlib.pyplot as plt
import sys

#same choices as PPPC4DMID
Nsec = 179
lgx = np.linspace(-8.9, 0, Nsec)
x = 10**lgx
Eprim  = 5e3
Esec = Eprim*x
dnde = dnde_pi0(Esec, Eprim)
dndlgx = dnde*Esec*np.log(10.)

#plt.loglog(x, dndlgx, '-b')

Eprim  = 160
Esec = Eprim*x
dnde = dnde_pi0(Esec, Eprim)
dndlgx = dnde*Esec*np.log(10.)

#plt.loglog(x, dndlgx, '-r')

dexo_pi = np.loadtxt('pi0_normed.dat')
#plt.loglog(dexo_pi[:,0], dexo_pi[:,2], '--k')
#print(np.trapz(dexo_pi[:,1], np.log10(dexo_pi[:,0])))


#try to reproduce
mpi0 = 0.13498 #GeV
def gamma(Eprim, mprim):
	return Eprim/mprim
def beta(Eprim, mprim):
	gam = gamma(Eprim, mprim)
	if gam < 1:
		print(Eprim, mprim, gam)
		sys.exit(0)
	return np.sqrt(1.-1./gam**2)
def dNdEpi0(Epi, Egamma):
	if Epi < mpi0:
		return 0.	
	gam = gamma(Epi, mpi0)
	bet = beta(Epi, mpi0)
	Eplus = mpi0/(2*gam*(1.-bet))
	Eminus = mpi0/(2*gam*(1.+bet))
	return 2./(gam*bet*mpi0)*(np.heaviside(Egamma-Eminus, 0.5)-np.heaviside(Egamma-Eplus, 0.5))

intGamma = integrate.quad(lambda Egamma: dNdEpi0(Eprim/1000., Egamma), 0, Eprim/1000.)[0]
dNdEtest = np.vectorize(dNdEpi0, excluded = ['Epi']).__call__(Eprim/1000., Esec/1000.)
dNdlgxtest = dNdEtest*Esec/1000.*np.log(10.)
#plt.loglog(x, dNdlgxtest, '--b')
#plt.show()
#sys.exit(0)


#produce outputs
Nprim = 20
Eprim = np.logspace(np.log10(0.1), np.log10(5), Nprim)
Eprims = []
Esecs = []
lgxs = []
for i in range(Nprim):
	Eprims = np.concatenate([Eprims, np.ones(Nsec)*Eprim[i]])
	Esecs = np.concatenate([Esecs, x*Eprim[i]])
	lgxs = np.concatenate([lgxs, lgx])

#print(np.stack((Eprims, Esecs)).T)
#dNdEpi0_gamma = np.vectorize(dNdEpi0).__call__(Eprims, Esecs)
#dNdlgxpi0_gamma = dNdEpi0_gamma*Esecs*np.log(10.)
#dNdlgxpi0_e = np.zeros(Nprim*Nsec)
#dout = np.stack((Eprims, lgxs, dNdlgxpi0_e, dNdlgxpi0_gamma)).T
#np.savetxt('secondariesfrompi0.dat', dout, fmt = '%1.6e')
#sys.exit()


#secondary from mu->e+nue+numu, use GeV from here
me = 0.511e-3
mmu = 0.10566
mpiCh = 0.13957
def dGammadECM(Ee):
	if Ee >= me and Ee <=mmu/2:
		return np.sqrt(Ee**2 - me**2)*(Ee*(mmu**2 + me**2 - 2*mmu*Ee) + 2*(Ee*mmu - me**2)*(mmu - Ee))
	else:
		return 0.

#y= cos(theta_Lab)
#def EeCM(Eelab, Eprim, mprim, y):
#	gam = gamma(Eprim, mprim)
#	bet = beta(Eprim, mprim)
#	return gam*Eelab*(1-bet*y)

def EeCM(Eelab, Eprim, mprim, y):
	gam = gamma(Eprim, mprim)
	bet = beta(Eprim, mprim)
	bete = beta(Eelab, me)	
	return gam*Eelab*(1+bet*bete*y)	

#overall normalization, this doesn't change with the reference frame
dNdEmu_norm = integrate.quad(dGammadECM, me, mmu/2)[0]

def dNdEmuCM(Emu, Eelab, y):
	return abs(dGammadECM(EeCM(Eelab, Emu, mmu, y))/dNdEmu_norm)

def dNdEmu(Emu, Eelab):
	if Emu < mmu:
		return 0.
	gam = gamma(Emu, mmu)
	bet = beta(Emu, mmu)
	if Eelab < me or Eelab > Emu:
		return 0.
	game = gamma(Eelab, me)
	bete = beta(Eelab, me)
	#prefactor = lambda y: 1/(2*gam*(bet*y-1.))
	prefactor = lambda y: bete*game/(2*np.sqrt(game**2*gam**2*(1+bet*bete*y)**2-1.))
	intcth = lambda y: prefactor(y)*dNdEmuCM(Emu, Eelab, y)
	#ymin = max([1/bet*(1.-mmu/(2*gam*Eelab)), -1])
	#ymax = min([1/bet*(1.-me/(gam*Eelab)), 1])
	ymax = min([1/(bet*bete)*(mmu/(2*gam*Eelab)-1.), 1])
	ymin = max([1/(bet*bete)*(me/(gam*Eelab)-1.), -1])		 
	#return integrate.quad(intcth, -1., 1.)[0]
	if ymin <= ymax:
		return abs(integrate.quad(intcth, ymin, ymax)[0])
	else:
		return 0.

"""
def dNdEmu_norm(Emu):
	gam = gamma(Emu, mmu)
	bet = beta(Emu, mmu)
	Eemin = max([me/gam/(1.+bet), me])
	Eemax = min([mmu/(2*gam*(1-bet)), Emu])
	return integrate.quad(lambda Eelab: dNdEmu(Emu, Eelab), Eemin, Eemax)[0]	
"""



Eprim = mmu
Esec = Eprim*x
dnde = np.vectorize(dGammadECM).__call__(Esec)/dNdEmu_norm
dndlgx = dnde*Esec*np.log(10.)

#plt.loglog(x, dndlgx, '-r')



Eprim = 5.
gam = gamma(Eprim, mmu)
bet = beta(Eprim, mmu)
Esec = Eprim*x
dnde = np.vectorize(dNdEmu, excluded = ['Emu']).__call__(Eprim, Esec)
dndlgx = dnde*Esec*np.log(10.)
#plt.loglog(x, dndlgx, '-b')

Eprim = 50.
gam = gamma(Eprim, mmu)
bet = beta(Eprim, mmu)
Esec = Eprim*x
dnde = np.vectorize(dNdEmu, excluded = ['Emu']).__call__(Eprim, Esec)
dndlgx = dnde*Esec*np.log(10.)
#plt.loglog(x, dndlgx, '--b')

dexo_mu = np.loadtxt('muon_normed.dat')
#plt.loglog(dexo_mu[:,0], dexo_mu[:,1], '--k')

#print(np.trapz(dexo_mu[:,1], np.log10(dexo_mu[:,0])))

#plt.show()


dNdEmu_e = np.vectorize(dNdEmu).__call__(Eprims, Esecs+me)
dNdlgxmu_e = dNdEmu_e*Esecs*np.log(10.)
#Input Egammas, Emus for hazma functions
dNdEmu_gamma = np.vectorize(dnde_mu).__call__(Esecs*1000, Eprims*1000.)
#print('test secondaries:')
#print(dnde_mu(Esecs*1000, Eprims[-1]*1000)*Esecs*1000*np.log(10))
#print(Esecs*1000)
dNdlgxmu_gamma = dNdEmu_gamma*Esecs*1000.*np.log(10.)
dout = np.stack((Eprims, lgxs, dNdlgxmu_e, dNdlgxmu_gamma)).T
np.savetxt('secondariesfrommuon.dat', dout, fmt = '%1.6e')


#secondary from pi decay pi->mu+nu_mu (e+nu_e is very subdominant), use GeV from here
def dNdEpitomu(Epi, Emulab):
	if Epi < mpiCh:
		return 0.
	gam = gamma(Epi, mpiCh)
	bet = beta(Epi, mpiCh)	
	E0 = (mpiCh**2+mmu**2)/(2*mpiCh)
	#Emumin = max([mmu, E0/(gam*(1.+bet))])
	#Emumax = min([E0/(gam*(1.-bet)), Epi])
	Emumin = max([mmu, gam*E0-bet*gam*np.sqrt(E0**2-mmu**2)])
	Emumax = min([Epi, gam*E0+bet*gam*np.sqrt(E0**2-mmu**2)])
	#print(E0/(gam*(1.+bet)), E0/(gam*(1.-bet)))
	#print(Emumin, Emumax)
	#print((Emumax-Emumin)/(2*bet*gam*E0))
	if Emulab >= Emumin and Emulab <=Emumax:
		#return 1./(2*bet*gam*E0)
		return 1./(2*bet*gam*np.sqrt(E0**2-mmu**2))
		#normalized probability
		#return 1./(Emumax-Emumin)
	else:
		return 0.

def dNdEpitoe(Epi, Eelab):
	if Epi < mpiCh:
		return 0.
	gam = gamma(Epi, mpiCh)
	bet = beta(Epi, mpiCh)	
	E0 = (mpiCh**2+mmu**2)/(2*mpiCh)
	Emumin = max([mmu, gam*E0-bet*gam*np.sqrt(E0**2-mmu**2)])
	Emumax = min([Epi, gam*E0+bet*gam*np.sqrt(E0**2-mmu**2)])
	#print(Emumin, Emumax)
	#Emumin = max([Emumin, Eelab*(mmu**2/(4*Eelab**2)+1)])
	#Emumax = min([Emumax, mmu/(2*me)*Eelab*(me**2/Eelab**2+1)])
	#Emumin = max([Emumin, mmu/(2*me**2)*(mmu*Eelab+np.sqrt((Eelab**2+me**2)*(mmu**2-4*me**2)))])
	#Emumax = min([Emumax, mmu/me*Eelab])
	#print(Emumin, Emumax)
	if Emumax <= Emumin:
	#if False:
		return 0.
	else:
		dNsq = lambda Emulab: dNdEpitomu(Epi, Emulab)*dNdEmu(Emulab, Eelab)
		#Emus = np.linspace(Emumin, Emumax, 1000)
		#intgd = np.vectorize(dNsq).__call__(Emus)
		#return np.trapz(intgd, Emus)
		return integrate.quad(dNsq, Emumin, Emumax)[0]
		#return integrate.quad(dNsq, mmu, Epi)[0]

#piCh decay at rest
def dNdEpitoeCM(Eelab):
	E0 = (mpiCh**2+mmu**2)/(2*mpiCh)
	return dNdEmu(E0, Eelab)
	

#Eprim = 0.2
#Esec = Eprim*x
#Esec = np.linspace(me, Eprim, 1000)
#dnde = np.vectorize(dNdEpitomu, excluded = ['Epi']).__call__(Eprim, Esec)
#dndlgx = dnde*Esec*np.log(10.)
#print(integrate.quad(lambda Emulab: dNdEpitomu(Eprim, Emulab), mmu, Eprim)[0])
#plt.loglog(Esec/Eprim, dndlgx, '--b')
#plt.show()
#sys.exit()


dNdEpiCh_e = np.vectorize(dNdEpitoe).__call__(Eprims, Esecs+me)
dNdlgxpiCh_e = dNdEpiCh_e*Esecs*np.log(10.)
#Input Egammas, Emus for hazma functions
dNdEpiCh_gamma = np.vectorize(dnde_pi).__call__(Esecs*1000, Eprims*1000.)
dNdlgxpiCh_gamma = dNdEpiCh_gamma*Esecs*1000.*np.log(10.)
dout = np.stack((Eprims, lgxs, dNdlgxpiCh_e, dNdlgxpiCh_gamma)).T
np.savetxt('secondariesfrompiCh.dat', dout, fmt = '%1.6e')
sys.exit()


Eprim = mpiCh+0.01
#Esec = Eprim*x
Esec = np.linspace(me, Eprim, 1000)
dnde = np.vectorize(dNdEpitoe, excluded = ['Epi']).__call__(Eprim, Esec)
dndlgx = dnde*Esec*np.log(10.)
plt.loglog((Esec-me)/Eprim, dndlgx, '--b')



Eprim = 5.
Esec = Eprim*x
#print(Esec)
dnde = np.vectorize(dNdEpitoe, excluded = ['Epi']).__call__(Eprim, Esec)
dndlgx = dnde*Esec*np.log(10.)
plt.loglog((Esec-me)/Eprim, dndlgx, '-b')

	
print(np.trapz(dndlgx, np.log10(Esec/Eprim)))
#print(integrate.quad(lambda Eelab: dNdEpitoe(Eprim, Eelab), me, Eprim)[0])

#print(integrate.quad(lambda Eelab: dNdEpitoe(mpiCh, Eelab), me, mpiCh)[0])

dexo_piCh = np.loadtxt('piCh_normed.dat')
plt.loglog(dexo_piCh[:,0], dexo_piCh[:,1], '--k')
print(np.trapz(dexo_piCh[:,1], np.log10(dexo_piCh[:,0])))

#Esec = Eprim*x
#dnde = np.vectorize(dNdEpitoe, excluded = ['Epi']).__call__(Eprim, Esec)
#dndlgx = dnde*Esec*np.log(10.)
#plt.loglog(x, dndlgx, '--b')

plt.xlabel('$x=E_e/E_\pi$')
plt.ylabel('$dN/dlog_{10}x$')
plt.legend(['$E_\pi=m_\pi$', '$E_\pi=5$ GeV', 'Exoclass'])	

plt.show()



