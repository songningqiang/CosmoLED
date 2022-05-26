import sys
sys.path.append('/PPPC4DMID/')

import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from scipy.interpolate import interp2d
from scipy.interpolate import interpn
from PPPC4DMID.spectrum_table_Dconvention import PBH_dMdt
from PPPC4DMID.spectrum_table_Dconvention import PBH_primary_spectrum
from PPPC4DMID.spectrum_table_Dconvention import get_temperature_from_mass
from PPPC4DMID.spectrum_table_Dconvention import get_mass_from_temperature
from PPPC4DMID.spectrum_table_Dconvention import get_radius_from_mass
from ctypes import *
import time

libodNoComptonTau = CDLL("./libodNoComptonTau.so")
libodNoComptonTau.od_dtdz.restype = c_double

libod = CDLL("./libod.so")
libod.od_dtdz.restype = c_double

pi = np.pi

ztSetup = False #indicates whether data for z(t) interpolation has been setup
spectrumSetup = False #indicates whether data for photon spectrum interpolation has been setup
tauSetup = False #indicates whether data for optical depth interpolation has been setup
icsSetup = False #indicates whether data for ICS interpolation has been setup
Tref_ics = 0.01 #Reference temperature used for ics calculating in eV

#Planck 2018
Omm = 0.3153
OmL = 0.6847
Omgamma = 5.38e-5
Omr = 1.68 * Omgamma #Include effect of neutrinos
h = 0.6736
Omdm = 0.266

#PDG2020
rhoc0_gcm3 = 1.87834e-29 * h**2 #critical density today in g/cm^3
T0_Kelvin = 2.7255 #Temperature today in kelvin
me = 510.9989461e3 #electron mass eV
mmu = 105.6583745e6 #muon mass eV
mpi0 = 134.9768e6 #neutral pion mass eV
mpipm = 139.57039e6 #charged pion mass eV
mproton = 9.382720813e8 #proton mass in eV
alphaEM = 1/137.035999084 #fine structure constant
Mpl = 1.22089e28  #4D Planck mass in eV
G_eV = 1/Mpl**2 #Gravitational constnat in eV^-2

LamQCD = 300e6 #QCD cutoff

#arxiv:1912.04296v2
#rhoEarth_GeVcm3 = 0.6 #DM density near earth in GeV/cm^3
#rNFW = 32 #Milky Way NFW fit radius kpc
#gNFW = 0.95 #Slope of Milky Way NFW profile
#https://iopscience.iop.org/article/10.1088/1475-7516/2019/10/037/pdf
rhoEarth_GeVcm3 = 0.3 #DM density near earth in GeV/cm^3
rNFW = 9 #Milky Way NFW fit radius kpc
gNFW = 1.2 #Slope of Milky Way NFW profile

#arxiv:1904.05721
#rEarth = 8.178 #kpc 
#arxiv:1807.09409
rEarth = 8.127 #kpc

##NFW values used in Siegert et al
#rNFW = 9.98
#rsum1 = rNFW + rEarth
#rhoEarth_GeVcm3 = 1.2341095 / 4 / (rEarth*np.power(rsum1,2)/(4*np.power(rNFW,3)))
#gNFW = 1
#h = .6736
#rhoc0_gcm3 = 9.1e-30
#Omdm = 0.2645
#OmL = 0.6847
#Omr = 0
#print(rhoEarth_GeVcm3)

#Physical Constants and Unit Conversions
sigmaT_eV = 1.70847747e-15 #Thompson cross-section in eV^-2
Mstar0 = 1.16e66 #eV/solarMass
eVperg = 5.6096e32 #eV/g
c = 2.99792458e8 #m/s
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


def calcDimensionRadius(Ned, Mstar):
    """Returns the size of the extra dimensions

    Parameters
    ----------
    Ned  : int
        Number of large extra dimensions
    Mstar: float
        Scale of quantum gravity in eV

    Returns
    -------
    R
        Size of the extra dimensions in eV^-1
    """
    R = np.power(Mpl / Mstar, 2./Ned) / Mstar / (2*pi)
    return R

def isMacroscopic(Mpbh, Ned, Mstar):
    """Returns whether BH is larger than the size of extra dimensions

    Parameters
    ----------
    Mpbh : float
        Mass of the blackhole in eV
    Ned  : int
        Number of large extra dimensions
    Mstar: float
        Scale of quantum gravity in eV

    Returns
    -------
    isMacroscopic : bool
        Boolean indicating whether black hole is larger than extra dimensions or not
    """

    R = calcDimensionRadius(Ned, Mstar)
    M_g = Mpbh/eVperg
    #rbh = get_radius_from_mass(M_g, Ned, Mstar*GeVpereV)*GeVpereV #bh radius in eV^-1
    rbh = 2*Mpbh / Mpl**2
    #print(R)
    #print(rbh)
    output = False
    if R < rbh:
        output = True
    return output

def H(z):
    """Returns the expansion rate of the universe at redshift z in eV

    Parameters
    ----------
    z : float or numpy array
        Redshift(s) to have the expansion rate evaluated at

    Returns
    -------
    same type as z
        Expansion rate in eV
    """
    return h * (100/kmperMpc * hbar) * np.sqrt(OmL + Omm*(1+z)**3 + Omr*(1+z)**4)

def setupzt():
    """Reads in data from file 'zvst.csv' and assigns the grid 
    to global variable tz to be used for interpolation

    Parameters
    ----------

    Returns
    -------
        
    """
    global tz, ztSetup
    print('setting up zoft')
    tz = np.loadtxt('zvst.csv',delimiter=',')
    ztSetup = True
    return

def setupTauInterpolation():
    """Reads in data from tauData folder and sets up
    function tauInterp. Interpolated function takes in Energy (eV)
    and redshift

    Parameters
    ----------

    Returns
    -------
        
    """
    global tauInterp, tauSetup
    print('setting up tau Interpolation')

    Es = np.loadtxt('tauData/Eaxis.txt',delimiter=',',skiprows=0)
    zs = np.loadtxt('tauData/zaxis.txt',delimiter=',',skiprows=0)
    tauData = np.loadtxt('tauData/tau IonCut6.txt',delimiter=',',skiprows=0).T

    tauInterp = interp2d(Es, zs, tauData)

    tauSetup = True
    return

def setupICSInterpolation():
    """Reads in data from ics folder and sets up
    functions ics_dNgamdEdt_thomsonT0_interp and ics_dNgamdEdt_relT0_interp.
    Interpolated function takes in electron and photon energy in eV
    and redshift

    Reads in grid values for electron energy loss and stores them in the numpy
    array ics_dNeddeltadt_grid and the axis information in ics_eEloss_axis

    Parameters
    ----------

    Returns
    -------
        
    """
    global ics_dNgamdEdt_thomsonT0_interp, ics_dNgamdEdt_relT0_interp, icsSetup, ics_secondarySpec_axis, ics_dNgamdE_grid 
    print('setting up ICS interpolation')

    #Read in Eloss axis info
    ics_secondarySpec_axis = np.loadtxt('icsData/icsSecondary_axis.csv',delimiter=',')

    #Read in Energy loss
    Ts = np.array([1e-4, 3e-4, 6e-4,
        1e-3, 3e-3, 6e-3,
        1e-2, 3e-2, 6e-2,
        1e-1, 3e-1, 6e-1])
    NT = Ts.size
    Ne = ics_secondarySpec_axis.shape[0]
    Ngam =ics_secondarySpec_axis.shape[0]

    ics_dNgamdE_grid = np.zeros([NT, Ne, Ngam])
    for i, T in enumerate(Ts):
        ics_dNgamdE_grid[i] = np.loadtxt('icsData/icsSecondary_dNgamdE%.0e_eV.csv'%T,delimiter=',')
                                                                                                                                                                
    icsSetup = True
    return

def setupSpectrumInterpolation():
    """Reads in data from file PPPC4DMID table and sets up
    function spectrumInterp that can be used for getting
    dNgamma/dtdE. Interpolated function takes in Energy
    and Temperature as its two functions (both in eV)

    Parameters
    ----------

    Returns
    -------
        
    """
    global photSpectrumInterps, elecSpectrumInterps, spectrumSetup
    print('setting up PPPC4DMID Interpolations')
    photSpectrumInterps = {}
    elecSpectrumInterps = {}
    for Ned in range(0,7):
        if Ned == 1:
            continue
        #print(Ned)
        axes = np.loadtxt('PPPC4DMID/secondaryspectral/DimopoulosConvention/%dd/axes.txt'%Ned,skiprows=1)
        logEs = axes[:,0]
        logTs = axes[:,1]
        photSpecData = np.loadtxt('PPPC4DMID/secondaryspectral/DimopoulosConvention/%dd/gamma.txt'%Ned,skiprows=0).T
        elecSpecData = np.loadtxt('PPPC4DMID/secondaryspectral/DimopoulosConvention/%dd/positron.txt'%Ned,skiprows=0).T
        
        photSpectrumInterps[Ned] = interp2d(logEs, logTs, photSpecData,'linear')
        elecSpectrumInterps[Ned] = interp2d(logEs, logTs, elecSpecData)

    spectrumSetup = True
    return

def zoft(t):
    """Returns the redshift when the the universe was age t measured in seconds.
    The output is not truly redshift but rather z where the redshift is 1+z

    Parameters
    ----------
    t : float or numpy array
        Age(s) of the universe measured in seconds

    Returns
    -------
    same type as t
        z: Redshift of the universe minus 1 
    """
    global tz
    if ztSetup:
        return np.interp(t,tz[:,0],tz[:,1])
    else:
        setupzt()
        return zoft(t)

def tofz(z):
    """Returns the age of the universe at redshift 1+z

    Parameters
    ----------
    z : float or numpy array
        Redshift minus 1

    Returns
    -------
    same type as z
        Age(s) of the universe measured in seconds
    """
    global tz
    if ztSetup:
        return np.interp(z,tz[:,1][::-1],tz[:,0][::-1])
    else:
        setupzt()
        return tofz(z)

def Thawk4D(Mpbh):
    """Calculated the Hawking temperature of a 4D blackhole

    Parameters
    ----------
    Mpbh : float or numpy array
        Mass of the blackhole in eV

    Returns
    -------
    same type as Mpbh
        Temperature of blackhole in eV. Returns -1 if the mass is zero or negative
    """
    T = 1 / (8*pi*GN*Mpbh)

    #if mass is less than or equal to zero emit nothing
    return np.where(Mpbh<=0,-1, T)

def dNgamdotdEpri4D(MpbhIn, E):
    """\\frac{dN_\\gamma}{dEdt}, the differential number
    of primary photons emitted by a 4D black hole per unit energy per unit time

    Parameters
    ----------
    Mpbh : float or numpy array that is the same length as E
        Mass of the blackhole in eV
    E    : numpy array
        Energy at which photon is emitted in eV

    Returns
    -------
    numpy array of same length as E
        Differential photon emission. This quantity is unitless.
        If Mpbh is a scalar then this is the energy spectrum from black hole
        with mass Mpbh. If Mpbh is an array then the $i^{th}$ element of the output
        corresponds to emission at energy E[i] from blackhole with mass Mpbh[i]
    """
    if Thawk4D(MpbhIn) > me:
        print('There should probably be secondaries from this 4D black hole')

    #This uses a fit for the spectrum valid until E>>T
    #Fit is from Ballesteros et al 2020 (1906.10113v2)
    M18 = MpbhIn/(1e18*eVperg)
    E0 = 6.54e-5 /GeVpereV
    denominator = np.power(M18*E/E0,-2.7) + np.power(M18*E/E0,6.7)
    return 2.5e21 *hbar *GeVpereV / denominator

    if isinstance(MpbhIn, np.ndarray):
        Mpbh = MpbhIn
    else:
        Mpbh = np.ones_like(E)*MpbhIn    

    T = Thawk4D(Mpbh) #Hawking temperature
    
    #This is the low E limit
    Q = 0 #No Charge
    a = 0 #Non-rotating
    rplus = Mpbh + (Mpbh**2 - Q**2 - a**2)**0.5
    k = 2*pi*T
    A = 4*pi * (rplus - Mpbh) / k
    sigma1 = 4.*A*(3*Mpbh**2 - a**2)*E**2 / 9.*G_eV**3

    #This is the high E limit (geometric limit)
    sigmag = 27*pi*G_eV**2 * Mpbh**2 #Absorption cross section

    #Mix between high E limit and low E limit
    r = (10*Mpbh*E*G_eV + E/T)/11
    sigmamix = r**2*sigmag + (1-r)**2*sigma1
    
    sigma = np.minimum.reduce([sigmamix, sigmag])

    #ratio = np.minimum(E/T, 100)
    ratio = E/T
    denominator = (np.exp(ratio) - 1)*2*pi**2
    numerator = E**2 * sigma

    #where there is no temperature emit nothing
    return np.where(T==-1,0,numerator/denominator)

def dNelecdotdESingleM(Mpbh, E, Ned, Mstar=10e12):
    """\\frac{dN_\\electron}{dEdt}, the differential number
    of electrons and positrons emitted per unit energy per unit time
    by a black hole with 4+Ned large dimensions and a mass of MpbhIn
    
    The case of 4D black holes only includes the primary photons and does
    not include secondary photons produced by heavy particles decaying.

    Parameters
    ----------
    Mpbh : float
        Mass of the blackhole in eV
    E    : numpy array
        Energy at which electrons is emitted in eV
    Ned  : int
        Number of large extra dimensions
    Mstar: float
        Scale of quantum gravity in eV        

    Returns
    -------
    numpy array of same length as E
        Differential electron and positron emission. This quantity is unitless.
        If Mpbh is a scalar then this is the energy spectrum from black hole
        with mass Mpbh. If Mpbh is an array then the $i^{th}$ element of the output
        corresponds to emission at energy E[i] from blackhole with mass Mpbh[i]
    """
    if Ned > 0 and not isMacroscopic(Mpbh,Ned,Mstar):
        #Setup interpolation if secondaries are included
        if Mstar != 10e12:
            print('WARNING: Table is only setup for Mstar = 10 TeV')

        if not spectrumSetup:
            setupSpectrumInterpolation()
        spectrumInterp = elecSpectrumInterps[Ned]

        if not isinstance(E, np.ndarray):
            E = np.ones(1)*E

        spectrum = np.zeros(E.shape[0])
        
        if Mpbh > 0:
            Mpbh_g = Mpbh/eVperg
            E_GeV = E*GeVpereV
            Mstar_GeV = Mstar*GeVpereV

            T_GeV = get_temperature_from_mass(Mpbh_g,Ned,Mstar_GeV)
            spectrum = spectrumInterp(np.log10(E_GeV),np.log10(T_GeV))/(np.log(10) * E_GeV)

        #multiply by 2 to include electrons and positrons
        return 2*spectrum
    #Treat as 4D black hole
    else:
        if not spectrumSetup:
            setupSpectrumInterpolation()
        spectrumInterp = elecSpectrumInterps[0]

        if not isinstance(E, np.ndarray):
            E = np.ones(1)*E

        spectrum = np.zeros(E.shape[0])
        
        if Mpbh > 0:
            Mpbh_g = Mpbh/eVperg
            E_GeV = E*GeVpereV
            Mstar_GeV = Mstar*GeVpereV

            T_GeV = Thawk4D(Mpbh) * GeVpereV
            spectrum = spectrumInterp(np.log10(E_GeV),np.log10(T_GeV))/(np.log(10) * E_GeV)

        #multiply by 2 to include electrons and positrons
        return 2*spectrum

def dNelecdotdE(MpbhIn, E, Ned, Mstar=10e12):
    """\\frac{d^2N_\\electron}{dEdt}, the differential number
    of electrons and positrons (combined) emitted by a black hole with 4+Ned large dimensions
    per unit energy per unit time
    
    Case of MpbhIn being a vector is not currenlty implemented.

    Parameters
    ----------
    MpbhIn : float or numpy array that is the same length as E
        Mass of the blackhole in eV
    E    : numpy array
        Energy at which electron is emitted in eV
    Ned  : int
        Number of large extra dimensions
    Mstar: float
        Scale of quantum gravity in eV    

    Returns
    -------
    numpy array of same length as E
        Differential electron emission. This quantity is unitless.
        If Mpbh is a scalar then this is the energy spectrum from black hole
        with mass Mpbh. If Mpbh is an array then the $i^{th}$ element of the output
        corresponds to emission at energy E[i] from blackhole with mass Mpbh[i]
    """
    if isinstance(MpbhIn, np.ndarray):
        print('Varying M not implemented for electron spectrum')
        return None
    else:
        return dNelecdotdESingleM(MpbhIn, E, Ned, Mstar)

def dNelecdt(MpbhIn, Ned, Mstar=10e12, NE = 10000):
    """Calculates the rate of production of electrons and positrons 
    from the evaporation of a black hole of mass MpbhIn in 4+Ned extra dimensions.  

    Parameters
    ----------
    MpbhIn : float
        Mass of the blackhole in eV
    Ned  : int
        Number of large extra dimensions
    Mstar: float
        Scale of quantum gravity in eV
    NE   : int
        Number of electron energies to integrate over        

    Returns
    -------
    float
        Rate of electron and positron production from black hole
        evaporation. For just positrons or just electrons divide by 2
    """
    Es = np.logspace(1,10,10000)

    dNelecdEdt =  dNelecdotdE(MpbhIn, Es, Ned, Mstar=10e12)

    return np.trapz(dNelecdEdt,Es)

def dNgamdotdESingleM(Mpbh, E, Ned, Mstar=10e12, primaryOnly=False):
    """\\frac{dN_\\gamma}{dEdt}, the differential number
    of photons emitted per unit energy per unit time
    by a black hole with 4+Ned large dimensions and a mass of MpbhIn

    Parameters
    ----------
    Mpbh : float
        Mass of the blackhole in eV
    E    : numpy array
        Energy at which photon is emitted in eV
    Ned  : int
        Number of large extra dimensions
    Mstar: float
        Scale of quantum gravity in eV
    primaryOnly: boolean
        If true, only include primary photons        

    Returns
    -------
    numpy array of same length as E
        Differential photon emission. This quantity is unitless.
        If Mpbh is a scalar then this is the energy spectrum from black hole
        with mass Mpbh. If Mpbh is an array then the $i^{th}$ element of the output
        corresponds to emission at energy E[i] from blackhole with mass Mpbh[i]
    """
    if Ned > 0 and not isMacroscopic(Mpbh,Ned,Mstar):
        #Setup interpolation if secondaries are included
        if not primaryOnly:
            if Mstar != 10e12:
                print('WARNING: Table is only setup for Mstar = 10 TeV')

            if not spectrumSetup:
                setupSpectrumInterpolation()
            spectrumInterp = photSpectrumInterps[Ned]

        if not isinstance(E, np.ndarray):
            E = np.ones(1)*E

        spectrum = np.zeros(E.shape[0])
        
        if Mpbh > 0:
            Mpbh_g = Mpbh/eVperg
            E_GeV = E*GeVpereV
            Mstar_GeV = Mstar*GeVpereV
            if primaryOnly:
                spectrum = PBH_primary_spectrum(E_GeV, Mpbh_g, 'gamma', Ned, Mstar_GeV) #Primary spectrum
            else:
                T_GeV = get_temperature_from_mass(Mpbh_g,Ned,Mstar_GeV)
                spectrum = spectrumInterp(np.log10(E_GeV),np.log10(T_GeV))/(np.log(10) * E_GeV)

        return spectrum

    #Treat as 4D black hole
    else:
        if not primaryOnly:
            if not spectrumSetup:
                setupSpectrumInterpolation()
            spectrumInterp = photSpectrumInterps[0]

        if not isinstance(E, np.ndarray):
            E = np.ones(1)*E

        spectrum = np.zeros(E.shape[0])
        
        if Mpbh > 0:
            Mpbh_g = Mpbh/eVperg
            E_GeV = E*GeVpereV

            if primaryOnly:
                spectrum = dNgamdotdEpri4D(Mpbh,E)
            else:
                T_GeV = Thawk4D(Mpbh)*GeVpereV
                spectrum = spectrumInterp(np.log10(E_GeV),np.log10(T_GeV))/(np.log(10) * E_GeV)

        return spectrum

def dNgamdotdEVaryingM(MpbhIn, E, Ned, Mstar=10e12, primaryOnly=False):
    """\\frac{dN_\\gamma}{dEdt}, the differential number
    of photons emitted by a black hole with 4+Ned large dimensions
    per unit energy per unit time
    
    The case of 4D black holes only includes the primary photons and does
    not include secondary photons produced by heavy particles decaying.

    MpbhIn and E must be numpy arrays of the same length. The ith cell of
    the output corresponds to the output at energy E[i] of black hole of mass
    MpbhIn[i]

    Parameters
    ----------
    MpbhIn : numpy array that is the same length as E
        Mass of the blackhole in eV
    E    : numpy array
        Energy at which photon is emitted in eV
    Ned  : int
        Number of large extra dimensions
    Mstar: float
        Scale of quantum gravity in eV
    primaryOnly: boolean
        If true, only include primary photons        

    Returns
    -------
    numpy array of same length as E
        Differential photon emission. This quantity is unitless.
        If Mpbh is a scalar then this is the energy spectrum from black hole
        with mass Mpbh. If Mpbh is an array then the $i^{th}$ element of the output
        corresponds to emission at energy E[i] from blackhole with mass Mpbh[i]
    """

    if isinstance(MpbhIn, np.ndarray):
        if MpbhIn.shape[0] != E.shape[0]:
            print('Error: Mpbh and E have incompatible shapes')
        Mpbh = np.copy(MpbhIn)
    elif isinstance(E, np.ndarray):
        Mpbh = np.ones(E.shape[0])*MpbhIn
    else:
        E = np.ones(1)*E
        Mpbh = np.ones(1)*MpbhIn

    spectrum = np.zeros(Mpbh.shape[0])-1
    for i, Mi in enumerate(Mpbh):
        spectrum[i] = dNgamdotdESingleM(Mpbh[i], E[i], Ned, Mstar, primaryOnly)

    return spectrum

def dNgamdotdE(MpbhIn, E, Ned, Mstar=10e12, primaryOnly=False):
    """\\frac{dN_\\gamma}{dEdt}, the differential number
    of photons emitted by a black hole with 4+Ned large dimensions
    per unit energy per unit time
    
    The case of 4D black holes only includes the primary photons and does
    not include secondary photons produced by heavy particles decaying.

    Parameters
    ----------
    MpbhIn : float or numpy array that is the same length as E
        Mass of the blackhole in eV
    E    : numpy array
        Energy at which photon is emitted in eV
    Ned  : int
        Number of large extra dimensions
    Mstar: float
        Scale of quantum gravity in eV
    primaryOnly: boolean
        If true, only include primary photons        

    Returns
    -------
    numpy array of same length as E
        Differential photon emission. This quantity is unitless.
        If Mpbh is a scalar then this is the energy spectrum from black hole
        with mass Mpbh. If Mpbh is an array then the $i^{th}$ element of the output
        corresponds to emission at energy E[i] from blackhole with mass Mpbh[i]
    """
    if isinstance(MpbhIn, np.ndarray):
        return dNgamdotdEVaryingM(MpbhIn, E, Ned, Mstar, primaryOnly)
    else:
        return dNgamdotdESingleM(MpbhIn, E, Ned, Mstar, primaryOnly)

def icsTotaldNgamdE(Egams, Ees, elecSpec, z):
    """Calcultes the differential photon spectrum $\\frac{dN_\\gamma}{dE}$, 
    produced by a spectrum of electrons cooling at redshift z.
    
    This is determined using interpolation tables produced via DarkHistory.

    Parameters
    ----------
    Egams   : numpy array
        Energy values of photon spectrum in eV
    Ees     : numpy array
        Energy values of electron spectrum in eV
    elecSpec: numpy array that is the same length as Ees
        Spectrum of electrons to be integrated over
    z       : float
        redshift (minus 1) that this process occurs at        

    Returns
    -------
    numpy array of same length as Egams
        Differential photon spectrum produced
    """
    
    if not icsSetup:
        setupICSInterpolation()

    T = T0_Kelvin * kB * (1+z) #CMB temperature in eV

    Ne = Ees.size
    Ngam = Egams.size
    icsSpec = np.zeros([Ngam, Ne])
    for i, E in enumerate(Ees):
        Ee = np.ones(Ngam)*E
        icsSpec[:,i] = icsdNgamdE(Egams, Ee, T) #photon spectrum from a single electron

    integrand = elecSpec*icsSpec
    return np.trapz(integrand,Ees, axis=1)

def positrondNgamdE(Egam):
    """Calcultes the differential photon spectrum $\\frac{dN_\\gamma}{dE}$, 
    produced by the annihilation of a single positron. 
    
    This assumes all positron form positronium with 1/4 going into the singlet
    state and 3/4 going into the triplet.

    Parameters
    ----------
    Egam   : float or numpy array
        Energy values of photon spectrum in eV    

    Returns
    -------
    float or numpy array of same length as Egams
        Differential photon spectrum produced
    """
    lineWidth = 1e3 #Width of annihilation line in eV
    dNgamdESinglet = 2*np.exp(-np.power((Egam-me)/lineWidth,2)/2)/(lineWidth * np.sqrt(2*pi))

    x = Egam/me
    dNgamdETriplet = (2-x)/x + x*(1-x)/np.power(2-x,2) 
    dNgamdETriplet += 2*np.log(1-x)*((1-x)/np.power(x,2) - np.power(1-x,2)/np.power(2-x,3))
    dNgamdETriplet *= (6/(pi**2 - 9)/me)
    dNgamdETriplet[Egam > me] = 0

    return dNgamdESinglet/4 + 3*dNgamdETriplet/4

def positrondNgamdotdE(MpbhIn, Egam, Ned, Mstar=10e12):
    """Calcultes the differential photon spectrum $\\frac{d^2N_\\gamma}{dEdt}$, 
    produced by the annihilation positrons that come from the evaporation of 
    a black hole of mass MpbhIn in 4+Ned extra dimensions. 
    
    This assumes all positron form positronium with 1/4 going into the singlet
    state and 3/4 going into the triplet.

    Parameters
    ----------
    MpbhIn : float
        Mass of black hole in eV
    Egam   : float or numpy array
        Energy values of photon spectrum in eV    
    Ned    : int
        Number of large extra dimensions
    Mstar  : float
        Reduced Planck Scale in eV

    Returns
    -------
    float or numpy array of same length as Egams
        Differential photon spectrum produced
    """
    #Factor of 1/2 included because dNelecdt includes electrons and positrons
    return positrondNgamdE(Egam) * dNelecdt(MpbhIn, Ned, Mstar)/2

def icsdNgamdE(Egam, Ee, T):
    """Calcultes the differential photon spectrum $\\frac{dN_\\gamma}{dE}$, 
    produced by a single electron of energy Ee. Outputs correspond to pairs
    of (Egam[i], Ee[i])
    
    This is determined using interpolation tables produced via DarkHistory.

    Parameters
    ----------
    Egams   : float or numpy array
        Energy values of photon spectrum in eV
    Ee      : same as Egams
        Energy of incoming electron in eV
    T       : float
        Temperature of the CMB in eV    

    Returns
    -------
    float or numpy array of same length as Egams
        Differential photon spectrum produced
    """
    if not icsSetup:
        setupICSInterpolation()

    axisEe = ics_secondarySpec_axis[:,0]
    axisEgam = ics_secondarySpec_axis[:,1]
    axisT = np.array([1e-4, 3e-4, 6e-4,
    1e-3, 3e-3, 6e-3,
    1e-2, 3e-2, 6e-2,
    1e-1, 3e-1, 6e-1])

    if isinstance(Ee, np.ndarray) and isinstance(Egam, np.ndarray):
        if Ee.shape != Egam.shape:
            print('Error: Ee and Egam must have the same shape')
        TVals = T*np.ones_like(Ee)
        return interpn((axisT, axisEe, axisEgam), ics_dNgamdE_grid, (TVals, Ee, Egam), bounds_error = False,fill_value = 0)
    else:
        if (not isinstance(Ee, np.ndarray)) or (not isinstance(Egam, np.ndarray)):
            print('Error: If one of Ee or Egam is an array, then both must be')
        return interpn((axisT, axisEe, axisEgam), ics_dNgamdE_grid, (T, Ee, Egam),bounds_error = False,fill_value = 0)    

def dMdt4D(t, M):
    """\\frac{dM}{dt}, change in 4D black hole mass in eV/s

    Parameters
    ----------
    t    : float
        Time (seconds) used as independent variable for integrator
    M    : float
        Mass of black hole in eV

    Returns
    -------
    float
        Instatneous mass loss in eV/s
    """
    if M < 0:
        return 0

    a = -9.43e123 #constant in eV^3/s
    T = Thawk4D(M)

    f = 1 #normalized to 1 for black holes of 10^17 g
    if T > me:
        f += 4*0.142
    if T > mmu:
        f += 4*0.142
    if T > mpi0:
        f += 0.267
    if T > mpipm:
        f += 2*0.267
    if T > LamQCD:
        f -= 3*0.267
        f += 3*12*0.142 #Light quarks
        f += 8*2*0.06 #gluons

    dmdt = a * f / M**2
    return dmdt

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
    if isMacroscopic(M, Ned, Mstar):
        return dMdt4D(t, M)

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

def bhDensity(fdm, M0):
    """The number density of black holes today if black holes of mass M0 (eV)
    make up the fraction fdm of the mass-energy of dark matter

    Parameters
    ----------
    fdm  : float
        Fraction of dark matter comprised of PBH
    M0   : float
        Mass of black hole in eV

    Returns
    -------
    float
        Number density of black holes in eV^3
    """
    rhoDM = Omdm * rhoc0_gcm3 *(hbar*c/mpercm)**3 * eVperg
    return fdm*rhoDM/M0

def gamDensity(fdm, zs, Ms, E, Ned, Mstar, M0, primaryOnly=False):
    """Calculates the number density of photons today
    at energy E (eV) per unit energy from black hole that has masses Ms (eV) at redshifts 1 + zs.
    
    This accounts for redshiffting of photons and absorption of photons as
    the only effects between the emission of photons and observation today.

    This determines the photon density by integrating over redshifts

    Parameters
    ----------
    fdm  : float
        Fraction of dark matter initially comprised of PBH
    zs   : numpy array
        List of redshifts (minus 1). Should be ordered from largest to smallest
    Ms   : numpy array (same size as zs)
        Mass of black holes at each value of zs (eV)
    E    : float
        Energy from the photon spectrum to be evaluated at
    Ned  : int
        Number of large extra dimensions. The default value is 0 for 4D case  
    Mstar: float
        Scale of gravity (eV). Irrelevant for 4D case but must be set in case
        with Ned > 0
    M0   : float
        Mass of the black holes when created (eV)
    primaryOnly: boolean
        If true, only include primary photons    

    Returns
    -------
    float
        Number density of photons today in eV^2
    """
    if not tauSetup:
        setupTauInterpolation()
    taus = np.flip(tauInterp(E, zs)[:,0])

    npbh = bhDensity(fdm, M0) #PBH number density today eV^3

    dndz = -npbh * dNgamdotdE(Ms, (1+zs)*E, Ned, Mstar, primaryOnly) / H(zs) * np.exp(-taus)
    #print('TAUS EXCLUDED')

    return np.trapz(dndz, zs)

def intensity(fdm, zs, Ms, E, Ned, Mstar, M0, primaryOnly=False):
    """Calculates the Intensity of photons at energy E (eV) today
     in $cm^{-2} s^{-1} sr^{-1}GeV^{-1}$ from black hole that has
     masses Ms (eV) at redshifts 1 + zs.
    
    This accounts for redshiffting of photons and absorption of photons as
    the only effects between the emission of photons and observation today.

    This determines the intensity by integrating over redshifts. See extraGalacticIntensity
    for a function that determiens the intensity by tracking the spectrum at each
    redshift

    Parameters
    ----------
    fdm  : float
        Fraction of dark matter initially comprised of PBH
    zs   : numpy array
        List of redshifts (minus 1). Should be ordered from largest to smallest
    Ms   : numpy array (same size as zs)
        Mass of black holes at each value of zs
    E    : float
        Energy from the photon spectrum to be evaluated at
    Ned  : int
        Number of large extra dimensions. The default value is 0 for 4D case  
    Mstar: float
        Scale of gravity (eV). Irrelevant for 4D case but must be set in case
        with Ned > 0
    primaryOnly: boolean
        If true, only include primary photons    
    Returns
    -------
    float
        Intensity of photons in $cm^{-2} s^{-1} sr^{-1} GeV^{-1}$
    """
    ngam = gamDensity(fdm, zs, Ms, E, Ned, Mstar, M0, primaryOnly) #gamma density in eV^2
    return ngam / (4*pi) / hbar**3 / c**2 * mpercm**2 / GeVpereV

def maxIntensityJohnson():
    """OLD!!! Outputs array with measured values of photon intensity.
    The data is digitized from a plot in arxiv:2005.07467v1

    Parameters
    ----------
    
    Returns
    -------
    2D numpy array
        First column - Energies in eV of the intensity measurements
        Second column- Intensity of photons in $cm^{-2} s^{-1} sr^{-1}$
    """
    egret = np.loadtxt('GammaData/EGRET.csv',delimiter=',')
    fermiLAT = np.loadtxt('GammaData/FERMI-LAT.csv',delimiter=',')
    comptel = np.loadtxt('GammaData/COMPTEL.csv', delimiter=',')
    moretti = np.loadtxt('GammaData/Moretti Fit.csv',delimiter=',')

    #combine energies into 1 column (data initialy in GeV)
    EsGeV = egret[:,0]
    EsGeV = np.append(EsGeV, fermiLAT[:,0])
    EsGeV = np.append(EsGeV, comptel[:,0])
    EsGeV = np.append(EsGeV, moretti[:,0])    
    
    Es = EsGeV*1e9 #eV

    #Combine intensitites into one column
    I = egret[:,1]
    I = np.append(I, fermiLAT[:,1])
    I = np.append(I, comptel[:,1])
    I = np.append(I, moretti[:,1])    
    
    return np.array([Es,I]).T

def maxIntensity(sigmaBound = 2, useAjello = True, plot=False):
    """Outputs array with measured values of photon intensity.

    Parameters
    ----------
    sigmaBound: float
        Number of standard deviations added to intensity values
    useAjello : boolean
        If true uses constraints compiled by Marco Ajello et al in 
        https://iopscience.iop.org/article/10.1086/592595/pdf
    Returns
    -------
    2D numpy array
        First column - Energies in eV of the intensity measurements
        Second column- Intensity of photons in $cm^{-2} s^{-1} sr^{-1} GeV^{-1}$.
        Thid column  - Minimum energy of each energy bin in eV
        Fourth column- Maximum energy of each energy bin in eV
    """
    if useAjello:
        #read all the data
        asca = np.loadtxt('AjelloData/ASCA.dat',delimiter=' ',skiprows=3)
        bat = np.loadtxt('AjelloData/BAT.dat',delimiter=' ',skiprows=3)
        comptel = np.loadtxt('AjelloData/Comptel.dat',delimiter=' ',skiprows=3)
        egret = np.loadtxt('AjelloData/EGRET.dat',delimiter=' ',skiprows=3)
        #graph = np.loadtxt('AjelloData/Graph.dat',delimiter=' ',skiprows=3)
        gruber = np.loadtxt('AjelloData/Gruber.dat',delimiter=' ',skiprows=3)
        heao1 = np.loadtxt('AjelloData/HEA01-A4_withEBins.dat',delimiter=',',skiprows=3)
        nagoya = np.loadtxt('AjelloData/Nagoya_withEBins.dat',delimiter=',',skiprows=3)
        smm = np.loadtxt('AjelloData/SMM_withEBins.dat',delimiter=',',skiprows=3)
        xte = np.loadtxt('AjelloData/XTE.dat',delimiter=' ',skiprows=3)
    
        if plot:
            plt.errorbar(1e3*asca[:,0], asca[:,3]/asca[:,0]**2/GeVperkeV, asca[:,5]/asca[:,0]**2/GeVperkeV, 1e3*asca[:,2],fmt='none', label='ASCA')
            plt.errorbar(1e3*bat[:,0], bat[:,3]/bat[:,0]**2/GeVperkeV, bat[:,5]/bat[:,0]**2/GeVperkeV, 1e3*bat[:,2],fmt='none', label='BAT')
            plt.errorbar(1e3*comptel[:,0], comptel[:,3]/comptel[:,0]**2/GeVperkeV, comptel[:,5]/comptel[:,0]**2/GeVperkeV, 1e3*comptel[:,2],fmt='none', label='COMPTEL')
            plt.errorbar(1e3*egret[:,0], egret[:,3]/egret[:,0]**2/GeVperkeV, egret[:,5]/egret[:,0]**2/GeVperkeV, 1e3*egret[:,2],fmt='none', label='EGRET')
            plt.errorbar(1e3*gruber[:,0], gruber[:,3]/gruber[:,0]**2/GeVperkeV, gruber[:,5]/gruber[:,0]**2/GeVperkeV, 1e3*gruber[:,2],fmt='none', label='HEAO-A4')
            plt.errorbar(1e3*heao1[:,0], heao1[:,3]/heao1[:,0]**2/GeVperkeV, heao1[:,5]/heao1[:,0]**2/GeVperkeV, 1e3*heao1[:,2],fmt='none', label='HEAO-1')
            plt.errorbar(1e3*nagoya[:,0], nagoya[:,3]/nagoya[:,0]**2/GeVperkeV, nagoya[:,5]/nagoya[:,0]**2/GeVperkeV, 1e3*nagoya[:,2],fmt='none', label='Nagoya')
            plt.errorbar(1e3*smm[:,0], smm[:,3]/smm[:,0]**2/GeVperkeV, smm[:,5]/smm[:,0]**2/GeVperkeV, 1e3*smm[:,2],fmt='none', label='SMM')
            plt.errorbar(1e3*xte[:,0], xte[:,3]/xte[:,0]**2/GeVperkeV, xte[:,5]/xte[:,0]**2/GeVperkeV, 1e3*xte[:,2],fmt='none', label='XTE')

            plt.ylim([0.1*np.min(egret[:,5]/egret[:,0]**2/GeVperkeV),100*np.max(asca[:,5]/asca[:,0]**2/GeVperkeV)])
            plt.xlim([0.5*np.min(1e3*asca[:,0]),2*np.max(1e3*egret[:,0])])
            #plt.legend(['ASCA','BAT','COMPTEL','EGRET','HEAO-A4','HEAO-1','Nagoya','SMM','XTE'],ncol=2,fontsize=12)

        #combine energies into one column
        EskeV = asca[:,0]
        EskeV = np.append(EskeV, bat[:,0])
        EskeV = np.append(EskeV, comptel[:,0])
        EskeV = np.append(EskeV, egret[:,0])
        #EskeV = np.append(EskeV, graph[:,0])
        EskeV = np.append(EskeV, gruber[:,0])
        EskeV = np.append(EskeV, heao1[:,0])
        EskeV = np.append(EskeV, nagoya[:,0])
        EskeV = np.append(EskeV, smm[:,0])
        EskeV = np.append(EskeV, xte[:,0])
        Es = EskeV*1e3 #eV

        #combine energy bin maxima into one column
        maxEskeV = asca[:,0]+asca[:,2]
        maxEskeV = np.append(maxEskeV, bat[:,0]+bat[:,2])
        maxEskeV = np.append(maxEskeV, comptel[:,0]+comptel[:,2])
        maxEskeV = np.append(maxEskeV, egret[:,0]+egret[:,2])
        #maxEskeV = np.append(maxEskeV, graph[:,0]+graph[:,2])
        maxEskeV = np.append(maxEskeV, gruber[:,0]+gruber[:,2])
        maxEskeV = np.append(maxEskeV, heao1[:,0]+heao1[:,2])
        maxEskeV = np.append(maxEskeV, nagoya[:,0]+nagoya[:,2])
        maxEskeV = np.append(maxEskeV, smm[:,0]+smm[:,2])
        maxEskeV = np.append(maxEskeV, xte[:,0]+xte[:,2])
        maxEs = maxEskeV*1e3 #eV

        #combine energy bin minima into one column
        minEskeV = asca[:,0]-asca[:,1]
        minEskeV = np.append(minEskeV, bat[:,0]-bat[:,1])
        minEskeV = np.append(minEskeV, comptel[:,0]-comptel[:,1])
        minEskeV = np.append(minEskeV, egret[:,0]-egret[:,1])
        #minEskeV = np.append(minEskeV, graph[:,0]-graph[:,1])
        minEskeV = np.append(minEskeV, gruber[:,0]-gruber[:,1])
        minEskeV = np.append(minEskeV, heao1[:,0]-heao1[:,1])
        minEskeV = np.append(minEskeV, nagoya[:,0]-nagoya[:,1])
        minEskeV = np.append(minEskeV, smm[:,0]-smm[:,1])
        minEskeV = np.append(minEskeV, xte[:,0]-xte[:,1])
        minEs = minEskeV*1e3 #eV

        #Combine intensitites into one column
        I = asca[:,3] + sigmaBound*asca[:,5]
        I = np.append(I, bat[:,3] + sigmaBound*bat[:,5] )
        I = np.append(I, comptel[:,3] + sigmaBound*comptel[:,5] )
        I = np.append(I, egret[:,3] + sigmaBound*egret[:,5] )
        #I = np.append(I, graph[:,3] + sigmaBound*graph[:,5] )
        I = np.append(I, gruber[:,3] + sigmaBound*gruber[:,5] )
        I = np.append(I, heao1[:,3] + sigmaBound*heao1[:,5] )
        I = np.append(I, nagoya[:,3] + sigmaBound*nagoya[:,5] )
        I = np.append(I, smm[:,3] + sigmaBound*smm[:,5] )
        I = np.append(I, xte[:,3] + sigmaBound*xte[:,5] )
        I = I / EskeV**2 / GeVperkeV

        EsOut = Es[Es.argsort()]
        minEsOut = minEs[Es.argsort()]
        maxEsOut = maxEs[Es.argsort()]
        IOut = I[Es.argsort()]

        return np.array([EsOut[maxEsOut < 1e10],IOut[maxEsOut < 1e10], minEsOut[maxEsOut < 1e10],maxEsOut[maxEsOut < 1e10]]).T
    else:
        chandraXMM = np.loadtxt('GammaData/Chandra_XMM-2004.csv',delimiter=',',skiprows=1)
        swift = np.loadtxt('GammaData/SWIFT_BAT-2008.csv',delimiter=',',skiprows=1)
        nagoya = np.loadtxt('GammaData/Nagoya-1975.csv',delimiter=',',skiprows=1)
        comptel = np.loadtxt('GammaData/COMPTEL-2000.csv',delimiter=',',skiprows=1)
        egret = np.loadtxt('GammaData/EGRET-2004.csv',delimiter=',',skiprows=1)
        fermiLAT = np.genfromtxt('GammaData/FERMI_LAT-2015.txt',skip_header=61,skip_footer=52)

        #combine energies into 1 column (data initialy in GeV)
        EsMeV = (egret[:,4]+egret[:,3])/2 
        EsMeV = np.append(EsMeV, (chandraXMM[:,4]+chandraXMM[:,3])/2)
        EsMeV = np.append(EsMeV, (swift[:,4]+swift[:,3])/2)
        EsMeV = np.append(EsMeV, (comptel[:,4]+comptel[:,3])/2)
        EsMeV = np.append(EsMeV, (fermiLAT[:,2] + fermiLAT[:,1])/2)
        EsMeV = np.append(EsMeV, (nagoya[:,4]+nagoya[:,3])/2)
        #print((swift[:,4]+swift[:,3])/2)
        Es = EsMeV*1e6 #eV

        #combine energy bin maximums into 1 column (data initialy in GeV)
        maxEsMeV = egret[:,4] 
        maxEsMeV = np.append(maxEsMeV, chandraXMM[:,4])
        maxEsMeV = np.append(maxEsMeV, swift[:,4])
        maxEsMeV = np.append(maxEsMeV, comptel[:,4])
        maxEsMeV = np.append(maxEsMeV, fermiLAT[:,2])
        maxEsMeV = np.append(maxEsMeV, nagoya[:,4])
        maxEs = maxEsMeV*1e6 #eV

        #combine energy bin minimums into 1 column (data initialy in GeV)
        minEsMeV = egret[:,3] 
        minEsMeV = np.append(minEsMeV, chandraXMM[:,3])
        minEsMeV = np.append(minEsMeV, swift[:,3])
        minEsMeV = np.append(minEsMeV, comptel[:,3])
        minEsMeV = np.append(minEsMeV, fermiLAT[:,1])
        minEsMeV = np.append(minEsMeV, nagoya[:,3])
        minEs = minEsMeV*1e6 #eV

        EmidFermiA = (fermiLAT[:,2] + fermiLAT[:,1])/2
        EwidthFermiA = (fermiLAT[:,2] - fermiLAT[:,1])
        IFermiA = fermiLAT[:,3]/EwidthFermiA
        dIFermiA = np.sqrt(fermiLAT[:,4]**2 + fermiLAT[:,6]**2)/EwidthFermiA 

        #Combine intensitites into one column
        I = (egret[:,1] + sigmaBound*egret[:,2]) 
        I = np.append(I, chandraXMM[:,1] + sigmaBound*chandraXMM[:,2])
        I = np.append(I, swift[:,1] + sigmaBound*swift[:,2])
        I = np.append(I, comptel[:,1] + sigmaBound*comptel[:,2])    
        I = np.append(I, IFermiA + sigmaBound*dIFermiA)
        I = np.append(I, nagoya[:,1] + sigmaBound*nagoya[:,2])    
        I = I / GeVperMeV
     
        EsOut = Es[Es.argsort()]
        minEsOut = minEs[Es.argsort()]
        maxEsOut = maxEs[Es.argsort()]
        IOut = I[Es.argsort()]

        return np.array([EsOut[maxEsOut < 1e10],IOut[maxEsOut < 1e10], minEsOut[maxEsOut < 1e10],maxEsOut[maxEsOut < 1e10]]).T

def maxfdm(M0, Ned, Mstar, includeICSSpec=True, includePositronSpec=True, includeGalactic=True, useAjello = True, comptonTreatment = 'Full', Nz = 5000):
    """Determines the constraint on fdm, the fraction of
    dark matter initially comprised of PBHs, for a monochromatic
    spectrum of mass M0 (eV) with Ned extra dimensions
    and a scale of gravity at Mstar (eV)

    Parameters
    ----------
    M0   : float
        Initial mass of black hole (eV)
    Ned  : int
        Number of large extra dimensions
    Mstar: float
        Scale of gravity (eV)
    includeICSSpec: boolean
        False if contribution from ICS should be ignored. Default is true
    includePositronSpec: boolean
        False if contribution from positron annihilation should be ignored. 
        Default is true.
    includeGalactic: boolean
        False if contribution of the galactic contribution to the isotropic
        photon spectrum should be ignored. Default is true.
    useAjello : boolean
        If true uses constraints compiled by Marco Ajello et al in 
        https://iopscience.iop.org/article/10.1086/592595/pdf. If false
        uses constraints compiled by me (Avi). Default is true
    comptonTreatment : string
        Sets which approximation is used for Compton scattering in extragalctic flux calculation:
            'Ignore'    - Compton scattering of photons is ignored
            'Attenuate' - It is assumed compton scattering absorbs photons
                          no downscattering included, included in optical depth.
                          Largely accurate for high energy photons if low energy products
                          are unimportant
            'FracEloss' - It is assumed that all photons of the same energy equally
                          scatter and lose a constant fraction of energy. Mostly true
                          for low Energy photons 
            'Full'      - Solves the full integro-differntial equation at each redshift step
                          Properly determines resultant spectrum but very slow
            'FullWhenEvaporated' - Uses the 'Full' calculation for black holes that have full
                          evaporated before today. Otherwise uses FracEloss
    Nz  : int
        Number of redshift steps

    Returns
    -------
    float
        maximum allowed value for fdm
    """
    zmin = 1e-2
    zmax = 1100 #decoupling

    zs = np.logspace(np.log10(zmax), np.log10(zmin), Nz) #redshifts
    ts = tofz(zs) #time in seconds

    Ms = calcMoft(ts, M0, Ned, Mstar, odeMethod = 'BDF') #Black hole masses in eV
    
    #If evaporating before CMB there is no constraint
    if Ms[0] == 0:
        return 1e100
    
    constraints = maxIntensity(useAjello=useAjello) #Observed signal that can not be exceeeded
    Ebins = constraints[:,0] #mid points of energy bins in eV
    Emins = constraints[:,2] #min values of energy bins in eV
    Emaxs = constraints[:,3] #max values of energy bins in eV
    binWidths = Emaxs-Emins
    Ilimit = constraints[:,1] #intensitites in s^-1 cm^-2 sr^-1 MeV^-1
    Nbins = Ebins.size

    Es = np.logspace(2, 10, 10000)

    #set the comptontreatment allowing to make the variation for 'FullWhenEvaporated'
    if comptonTreatment == 'FullWhenEvaporated':
        if Ms[-1] > 0:
            usedComptonTreatment = 'FracEloss'
        else:
            usedComptonTreatment = 'Full'
    else:
        usedComptonTreatment = comptonTreatment

    If1 = extraGalacticIntensity(1, zs, Ms, Es, Ned, Mstar, M0, includeICS = includeICSSpec, includePositron = includePositronSpec, comptonTreatment = usedComptonTreatment) #intensity with fdm = 1

    if Ms[-1] > 0 and includeGalactic:
        rhoEarth_eV = rhoEarth_GeVcm3 / GeVpereV * (hbar*c/mpercm)**3 #eV^4
        rNFW_eV = rNFW / kpcPerMpc / eVMpc #eV
        rEarth_eV = rEarth / kpcPerMpc / eVMpc #eV

        galI = galacticIntensity(Es, Ms[-1]/M0, Ms[-1], Ned, Mstar, rhoEarth_eV,rNFW_eV,rEarth_eV,gNFW, includePositron = includePositronSpec)
        If1 += galI

    #if nothing is emitted then there is no constraint
    if np.amax(If1) <= 0:
        print('NO EMISSION')
        return 1e100

    If1Calc = np.copy(If1)

    #average Intensity over bin energy range

    obsIf1 = [np.trapz(If1[np.all(np.array([Es>Emins[i], Es < Emaxs[i]]),axis=0)], Es[np.all(np.array([Es>Emins[i], Es < Emaxs[i]]),axis=0)]) for i in range(Emins.size)]
    obsIf1 /= binWidths

#    index = (np.where(Es > 511e3))[0]
#    print(Es[index[0]])
#    print(If1[index[0]])
#    print(If1[index[0]-1])
#    print(If1[index[0]+1])
#    Ipeak = If1[index[0] -1]
#    index2 = (np.where(Ebins > 511e3))[0]

#    obsIf1 = np.interp(Ebins, Es, If1)    
#    obsIf1[index2[0]] = Ipeak

    If1 = obsIf1
    ratios = If1/Ilimit

    plt.loglog(Ebins, If1/np.max(ratios),'b.')

    #np.savetxt('N%d M%e spectrum.txt',np.array([Es,If1Calc]).T)

    return 1/np.max(ratios)

def redshiftPhotSpec(EsIn, specIn, rsInit, rsFin):
    """Redshifts a differntial spectrum of photons (dN/dE)

    Parameters
    ----------
    EsIn   : numpy array
        Energy values of the spectrum
    specIn : numpy array
        dN/dE values at redshift rsInit (must be same length as EsIn)
    rsInit : float
        Initial redshift value
    rsFin  : float
        Value to redshift the spectrum to

    Returns
    -------
    numpy array
        Resultant differential spectrum
    """
    rsEs = EsIn* (rsFin/rsInit)
    rsSpec = specIn * (rsInit/rsFin) #From sredshifting 1/dE
    rsSpec *= np.power(rsFin/rsInit, 3) #From density change in dN
    return np.interp(EsIn, rsEs, rsSpec,right=0)

def dNdEtoN(Es, dNdE):
    """Converts a differential spectrum dN/dE to a total spectrum N(E)

    Parameters
    ----------
    Es   : numpy array
        Energy bins
    dNdE : numpy array
        Differential spectrum must be in inverse the units of Es

    Returns
    -------
    numpy array
        Total spectrum N(E)
    """
    dE = np.gradient(Es)
    return dNdE*dE

def NtodNdE(Es, N):
    """Converts a total spectrum N to a differential spectrum dN/dE

    Parameters
    ----------
    Es   : numpy array
        Energy bins
    N    : numpy array
        Total spectrum

    Returns
    -------
    numpy array
        differential spectrum in the inverse units of Es
    """
    dE = np.gradient(Es)
    return N/dE

def dTau(E, z, dz, includeCompton=False):
    """calculates the differential photon absorption $\\frac{d\\tau}{dz}\delta z$
    at a given redshift

    Parameters
    ----------
    E  : numpy array or float
        Energy or energies of photons
    z  : float
        Redshift minus one at which the absorption happens
    dz : float
        Change in redshift that absorption happens over. This should not be large.
    includeCompton: boolean
        Whether to include Compton scattering in attenuation calculation
    Returns
    -------
    numpy array or float
        Differential absorption rate over a redshift step
    """
    if type(E) == np.ndarray:
        #dtdl = np.zeros(E.size)
        NE = E.size
        cE = E.ctypes.data_as(POINTER(c_double))
        cdtdzs = np.zeros(NE,dtype=np.float64).ctypes.data_as(POINTER(c_double))
        if includeCompton:
            libod.od_dtdzVec(c_double(z), cE, cdtdzs, NE)
        else:
            libodNoComptonTau.od_dtdzVec(c_double(z), cE, cdtdzs, NE)
        dtdz = np.ctypeslib.as_array(cdtdzs, E.shape)
    else:
        if includeCompton:
            dtdz = libod.od_dtdz(c_double(z), c_double(E))
        else:
            dtdz = libodNoComptonTau.od_dtdz(c_double(z), c_double(E))
    return dtdz*dz

def photEngLossCompton(E, dNdEIn, z, dz):
    """Transforms a differntial spectrum of photons (dN/dE)
    to account for the energy loss the photons experience due
    to Compton scattering at redshift z over a step dz.

    Parameters
    ----------
    E      : numpy array
        Energy values of the spectrum
    dNdEIn : numpy array
        dN/dE values at start of step (must be same length as EsIn)
    z      : float
        Initial redshift value
    dz     : float
        Length of redshift step

    Returns
    -------
    numpy array
        Resultant differential spectrum
    """
    NE = E.size
    cE = E.ctypes.data_as(POINTER(c_double))
    cdEdzs = np.zeros(NE,dtype=np.float64).ctypes.data_as(POINTER(c_double))
    libod.od_dEdzComptonVec(c_double(z), cE, cdEdzs, NE)
    dEdzs = np.ctypeslib.as_array(cdEdzs, E.shape)

    dEs = dEdzs*dz
    shiftEs = E - dEs

    N = dNdEtoN(E, dNdEIn)
    
    shifteddNdE = NtodNdE(shiftEs, N)
    return np.interp(E, shiftEs, shifteddNdE)

def dsigmadcosthCompton(costh, Ein):
    """Differential Compton cross section in the rest frame of the electron for an incoming
    photon of energy Ein scattring at with an angle of $\cos^{-1}$(costh).

    Parameters
    ----------
    costh      : float or numpy array
        cosine of the angle of scattering
    Ein        : float or numpy array
        Energy of incoming photon in eV, must be the same length as costh if they are arrays 

    Returns
    -------
    float or numpy array
        differential cross section in eV^-2
    """
    sin2th = 1 - costh**2

    Eout = Ein / (1 + Ein*(1-costh)/me)

    dsigmadcosth = pi*alphaEM**2/me**2
    dsigmadcosth *= (Eout/Ein)**2
    dsigmadcosth *= (Eout/Ein + Ein/Eout - sin2th)
    return dsigmadcosth

def sigmaCompton(E):
    """Total Compton cross section in the rest frame of the electron for an incoming
    photon of energy E. Formula from Ribicki and Lightman Radiative Processes in Astrophysics

    Parameters
    ----------
    E        : float or numpy array
        Energy of incoming photon in eV

    Returns
    -------
    float or numpy array
        Compton cross section in eV^-2
    """
    x = E/me

    sigma = (1+x)/np.power(x,3)
    sigma *= (2*x*(1+x)/(1+2*x) - np.log(1+2*x))

    sigma += np.log(1+2*x)/(2*x)
    sigma += -(1+3*x)/np.power(1+2*x,2)

    sigma *= 0.75*sigmaT_eV

    return sigma

def dNgamdotdECompton(E, dNgamdE, z):
    """Calculates the instataneous rate of change of a flux of photons due to
    Compton scattering at reshift z. This solves the equation:
    $\\frac{dN}{dEdt} = -n_e(z)\sigma(E)\\frac{dN}{dE}(E) + \\frac{n_e m_e}{E^2}\int dE' \\frac{d\sigma(E')}{d\cos (\\theta)} \\frac{dN}{dE'}(E')$
    The first term is a "loss" term corresponding to photons in an energy bin scattering.
    The second term is a "gain" term corresponding to the photons entering an energy bin
    due to scattering from a higher energy bin.

    Parameters
    ----------
    E    : numpy array
        Energy indices in eV
    dNgamdE : numpy array
        Flux of incoming photons at energies E in eV^2
    z   : float
        Redshift - 1

    Returns
    -------
    numpy array
        instantaneous Change in flux in eV^3
    """
    loss = sigmaCompton(E)*dNgamdE #Loss term without n_e

    #Function for interpolating the internal flux. Assumes flux is zero outside E range
    def fluxOfE(Ei):
        if Ei > np.max(E):
            return 0
        else:
            return np.interp(Ei, E, dNgamdE) 

    #Integrand in the flux term in terms of Ein (incoming energy) and E (outgoing energy)
    def dgaindEi(Ein, E):
        costh = 1 - (Ein/E - 1)*me/Ein
        return me/E**2 * dsigmadcosthCompton(costh, Ein) * fluxOfE(Ein) 

    #Set bounds of integration
    Eimin = E

    EMAX = E[-1]
    Eimax = np.zeros_like(E)
    Eimax[E >= me/2] = EMAX
    Eimax[E < me/2] = E[E < me/2]/(1 - 2*E[E < me/2]/me)    
    Eimax[Eimax > EMAX] = EMAX

    #Integrate to determine the total gian in each bin from the source term
    gain = np.array([integrate.quad(dgaindEi, Eimin[i], Eimax[i], args=(E[i]),points=np.logspace(np.log10(Eimin[i]), np.log10(Eimax[i]),10))[0] for i in range(len(E))])
    
    #Calculate the density of electrons in the universe (treats all electrons as free)
    nb0_cm3 = (Omm-Omdm)*rhoc0_gcm3/(mproton/eVperg)
    nElec = nb0_cm3 * (1+z)**3 / mpercm**3 * (c*hbar)**3 

    return nElec*(gain - loss)

def extraGalacticIntensity(fdm, zs, Ms, Es, Ned, Mstar, M0, primaryOnly = False, comptonTreatment = 'Full', includeICS=True, includePositron=True, specToday=True, NEgam = 10000, NEelec = 100, NEgamICS= 150, NEgamCompton=150):
    """Calculates the Intensity of photons at energy E (eV) today
     in $cm^{-2} s^{-1} sr^{-1}GeV^{-1}$ from black hole that has 
    masses Ms (eV) at redshifts 1 + zs.
    
    The intesnity is calculated by tracking the new spectrum injected
    for each redshift step and tracking how the spectrum changes due to
    absorption, redshifting, Compton scattering.

    Parameters
    ----------
    fdm  : float
        Fraction of dark matter initially comprised of PBH
    zs   : numpy array
        List of redshifts (minus 1). Should be ordered from largest to smallest
    Ms   : numpy array (same size as zs)
        Mass of black holes at each value of zs
    Es   : numpy array
        Energies for the photon spectrum to be evaluated at. All values must be between
        100 eV and 10 GeV.
    Ned  : int
        Number of large extra dimensions. The default value is 0 for 4D case  
    Mstar: float
        Scale of gravity (eV). Irrelevant for 4D case but must be set in case
        with Ned > 0
    primaryOnly: boolean
        If true, only include primary photons. Default is false
    comptonTreatment: string
        Determines which approximation is used for Compton scattering:
            'Ignore'    - Compton scattering of photons is ignored
            'Attenuate' - It is assumed compton scattering absorbs photons
                          no downscattering included, included in optical depth.
                          Largely accurate for high energy photons if low energy products
                          are unimportant
            'FracEloss' - It is assumed that all photons of the same energy equally
                          scatter and lose a constant fraction of energy. Mostly true
                          for low Energy photons 
            'Full'      - Solves the full integro-differntial equation at each redshift step
                          Properly determines resultant spectrum but very slow. If the redshift
                          steps are too long then this will no longer be accurate. Assumption is
                          photons can only scatter once within each step, this is true when
                          $abs(n_e(z) \\sigma(E) \Delta t) << 1$. If that is larger than 1, 
                          an exception is raised.
    includeICS: boolean
        If true, secondary photon spectrum from inverse Compton scattering of electrons
        is included. Default is true.
    includePositron: boolean
        If true, photon spectrum from positron annihilation is included. 
        Default is true.
    specToday : boolean
        If true, the final spectrum is redshifted to z=0. Otherwise the final spectrum
        is returned at the final value in zs. Default is true/
    NEgam : int
        Size of photon spectrum grid during calculation. Default is 10000
    NEelec : int
        Size of electron spectrum grid during ICS calculation. Default is 100
    NEgamICS: int
        Size of photon spectrum grid during ICS integration. Default is 150
    NEgamCompton: int
        Size of photon spectrum grid used when solving integro-differential equation
        for Compton scattering. Only relevant if comptonTreatment='Full'. Default is 150
    Returns
    -------
    numpy array
        Differential intensity of photons in $cm^{-2} s^{-1} sr^{-1} GeV^{-1}$ 
        for each energy in Es
    """
    print('NEgamcompton = %d'%NEgamCompton)
    print('NEgam = %d'%NEgam)

    if comptonTreatment != 'Ignore' and \
        comptonTreatment != 'Attenuate' and \
        comptonTreatment != 'FracEloss' and \
        comptonTreatment != 'Full':
            raise Exception('Compton approximation "' + comptonTreatment + '" is not recognized')

    redshifts = zs+1

    #setup grids
    Egams = np.logspace(2, 10, NEgam)
    EgamsICS = np.logspace(2, 10, NEgamICS)
    EgamsCompton = np.logspace(2, 10, NEgamCompton)
    Eelecs = np.logspace(2, 10, NEelec)

    photSpec = np.zeros(NEgam)
    if includeICS:
        icsPhotSpec = np.zeros(NEgam)

    npbh0 = bhDensity(fdm, M0) #PBH number density today eV^3

    #Loop through redshifts
    for i, rs in enumerate(redshifts):
        #print('rs: %f  '%(rs) + str(isMacroscopic(Ms[i], Ned, Mstar)))

#        if i % 100 == 2:
#            plt.loglog(Egams, photSpec)
#            plt.ylim([photSpec[0] * 1e-2, np.max(photSpec) * 1e1])
#            plt.show()

        if i == 0:
            continue

        npbh = npbh0 * rs**3 #PBH number density at redshift rs eV^3

        #redshift photon spectrum
        photSpec = redshiftPhotSpec(Egams, photSpec,redshifts[i-1],rs)

        #absorb photons
        if comptonTreatment == 'Attenuate':
            dTaus = dTau(Egams, rs-1, redshifts[i-1]-rs, includeCompton=True)
        else:
            dTaus = dTau(Egams, rs-1, redshifts[i-1]-rs, includeCompton=False)

        photSpec *= np.exp(-dTaus)

        #photon downscattering from Compton scattering
        if comptonTreatment == 'FracEloss':
            photSpec = photEngLossCompton(Egams, photSpec, rs-1, redshifts[i-1]-rs)
        elif comptonTreatment == 'Full':
            #Interpolate spectrum to sparse grid. Ensure interpolation does create negative flux
            dNdEFit = np.interp(EgamsCompton, Egams, photSpec)
            dNdEFit[dNdEFit < 0] = 0

            dNdEdt = dNgamdotdECompton(EgamsCompton, dNdEFit, rs-1)
            #Interpoalte back to full grid
            dNdEdtFit = np.interp(Egams, EgamsCompton, dNdEdt)

            dt = 1/H(rs-1)/rs * (redshifts[i-1] - rs)

            photSpec += dNdEdtFit*dt

            #If photon flux is now negative either interpolation has gone wrong (which can be fixed)
            #or time step is too long
            if np.min(photSpec) < 0:
                #Check if photSpec is zero only due to interpolation back to full grid
                #If flux is changed on sparse grid and found to still be positive then
                #the flux is only negative due to interpolation and negative flux points
                #set to zero. If sparse grid flux is also negative than calculation found
                #more photons scatter in an energy bin than exist and time step is too long

                dNdEFit += dNdEdt*dt #updated flux on sparse grid
                if np.min(dNdEFit) < 0:
                    print('i = %d\nrs=%f\n'%(i,rs))
                    print('E with dNdEFit < 0:')
                    print(EgamsCompton[dNdEFit < 0])
                    
                    raise Exception('Spectrum went negative, time step is too long')
                #if photSpec is negative only due to interpolation, force min to 0
                else:
                    photSpec[photSpec < 0] = 0

        #add new spectrum
        injPhotSpec = dNgamdotdE(Ms[i], Egams, Ned, Mstar, primaryOnly)
        if includePositron:
            injPhotSpec += positrondNgamdotdE(Ms[i], Egams, Ned, Mstar)

        photSpec += npbh*injPhotSpec/H(rs-1)/rs * (redshifts[i-1] - rs) 

        #repeat for ICS spectrum if it included. See above comments for explanation of equivilant section
        if includeICS:
            icsPhotSpec = redshiftPhotSpec(Egams, icsPhotSpec,redshifts[i-1],rs)
            icsPhotSpec *= np.exp(-dTaus)
            if comptonTreatment == 'FracEloss':
                icsPhotSpec = photEngLossCompton(Egams, icsPhotSpec, rs-1, redshifts[i-1]-rs)
            elif comptonTreatment == 'Full':
                dNdEFit = np.interp(EgamsCompton, Egams, icsPhotSpec)
                dNdEdt = dNgamdotdECompton(EgamsCompton, dNdEFit, rs-1)
                dt = 1/H(rs-1)/rs * (redshifts[i-1] - rs) 

                dNdEdtFit = np.interp(Egams, EgamsCompton, dNdEdt)
                icsPhotSpec += dNdEdtFit*dt

                if np.min(icsPhotSpec) < 0:
                    dNdEFit += dNdEdt*dt #updated flux on sparse grid
                    if np.min(dNdEFit) < 0:
                        print('i = %d\nrs=%f\n'%(i,rs))
                        print('E with dNdEFit < 0:')
                        print(EgamsCompton[dNdEFit < 0])
                        
                        raise Exception('ICS Spectrum went negative, time step is too long')
                    else:
                        icsPhotSpec[icsPhotSpec < 0] = 0

            #only add ICS spectrum if BH temperature is above 0.1*electron mass
            Mpbh_g = Ms[i] / eVperg
            Mstar_GeV = Mstar*GeVpereV
            T_GeV = get_temperature_from_mass(Mpbh_g,Ned,Mstar_GeV)
            T = T_GeV / GeVpereV
            if T > me/10:
                injElecSpec = dNelecdotdE(Ms[i], Eelecs, Ned, Mstar)
                elecSpec = npbh*injElecSpec/H(rs-1)/rs * (redshifts[i-1] - rs) 
                injICSPhotSpec = icsTotaldNgamdE(EgamsICS, Eelecs, elecSpec, rs-1)
                icsPhotSpec += np.interp(Egams, EgamsICS, injICSPhotSpec)

    #redshift to z=0 if option is chosen
    if specToday:
        photSpec = redshiftPhotSpec(Egams, photSpec,redshifts[-1],1)

    photSpecOut = np.interp(Es, Egams, photSpec)
    if includeICS:
        if specToday:
            icsPhotSpec = redshiftPhotSpec(Egams, icsPhotSpec,redshifts[-1],1)
        photSpecOut += np.interp(Es, Egams, icsPhotSpec)

    intensity = photSpecOut / (4*pi) / hbar**3 / c**2 * mpercm**2 / GeVpereV
    return intensity

def galacticIntensity(Es, fdm0, M, Ned, Mstar, rhoEarth_eV, rNFW_eV, rEarth_eV, gNFW, primaryOnly=False, includePositron=True):
    """Calculates the isotropic Intensity of photons at energy E (eV) today
     in $cm^{-2} s^{-1} sr^{-1}GeV^{-1}$ from black hole of mass M
    within the Milky Way halo.

    This assumes that the dark matter halo obeys the the gNFW profile of
    $\\rho(r) = \\rho_{earth}(\\frac{r_{earth}}{r})^g (\\frac{r_s + r_{earth}}{r_s + r})^{3-g}$.
    The isotropic signal is taken to be the signal from the line of sight oppsoite to the
    galactic centre.

    Parameters
    ----------
    Es   : numpy array
        Energies for the photon spectrum to be evaluated at. All values must be between
        100 eV and 10 GeV. They must also be ordered smallest to largest
    fdm  : float
        Fraction of dark matter comprised of PBH today
    M    : float
        Mass of black holes in eV
    Ned  : int
        Number of large extra dimensions. The default value is 0 for 4D case  
    Mstar: float
        Scale of gravity (eV). Irrelevant for 4D case but must be set in case
        with Ned > 0
    rhoEarth_eV: float
        Density of dark matter in the area of earth in eV^4
    rNFW_eV: float
        Radius parameter, $\\r_s$, in the NFW profile in eV^-1
    rEarth_eV: float
        Distance of Earth to the galactic centre in eV^-1
    gNFW: float
        Slope parameter in generalized NFW profile
    primaryOnly: boolean
        If true, only include primary photons. Default is false
    includePositron: boolean
        If true, photon spectrum from positron annihilation is included. 
        Default is true.
    
    Returns
    -------
    numpy array
        Differential intensity of photons in $cm^{-2} s^{-1} sr^{-1} GeV^{-1}$ 
        for each energy in Es
    """
    injSpec = dNgamdotdE(M, Es, Ned, Mstar, primaryOnly) #dimmensionless
    if includePositron:
        injSpec += positrondNgamdotdE(M, Es, Ned, Mstar)

    rsum = rEarth_eV + rNFW_eV
    #Calculate integral over line of sight away from galactic centre
    #if g=1 fully general analytic solution is indeterminate
    if abs(gNFW - 1) <= 1e-10:
        x = rEarth_eV/rNFW_eV
        rhoNFW_eV = rhoEarth_eV*rEarth_eV*np.power(rsum,2)/(4*np.power(rNFW_eV,3))
        Dmin = 4*(np.log((1+x)/x) - 1/(1+x)) * rhoNFW_eV*rNFW_eV #eV^3
    else:
        Dmin = rhoEarth_eV * np.power(1/rNFW_eV,2+gNFW) * np.power(rNFW_eV/rsum,gNFW)*rsum* (np.power(rEarth_eV,gNFW)*np.power(rsum,2) - rEarth_eV*np.power(rsum,gNFW)*(rEarth_eV -(gNFW-2)*rNFW_eV)) / (gNFW-2)/(gNFW-1)

    ngam = fdm0/M * injSpec * Dmin #eV^2
    return ngam / (4*pi) / hbar**3 / c**2 * mpercm**2 / GeVpereV

def lifetime(Mbh, Ned):
    Nz = 5000
    #zmin = 1e-2
    #zmax = 1100 #decoupling

    #zs = np.logspace(np.log10(zmax), np.log10(zmin), Nz) #redshifts
    #ts = tofz(zs) #time in seconds
    ts = np.logspace(6,18,10000)
    Ms = calcMoft(ts, Mbh, Ned, 10e12, odeMethod = 'BDF') #Black hole masses in eV
    plt.loglog(ts, Ms/eVperg)
    #print(Ms[-1])
    return ts[Ms > 0][-1]

def main():
#    Ned = 0
#    M_g = 1.2e17

#    #M_g = 47639.3801040134
#    #M_g = 3e5
#    M = M_g*eVperg

#    Mstar = 10e12
#    
#    compton = "Ignore"
#    maxf = maxfdm(M, Ned, Mstar, includeICSSpec = False, includeGalactic =True, includePositronSpec=True, useAjello=True, comptonTreatment = compton)
#    Nz = 5000
#    zmin = 1e-2
#    zmax = 1100 #decoupling

#    print('fdm = %.2e'%maxf)
#    
#    zs = np.logspace(np.log10(zmax), np.log10(zmin), Nz) #redshifts
#    ts = tofz(zs) #time in seconds
#    
#    Ms = calcMoft(ts, M, Ned, Mstar, odeMethod = 'BDF') #Black hole masses in eV
#    print('Mfin %.2e'%(Ms[-1]/eVperg))
#    
#    Es = np.logspace(2,10,10000)
#    Is = extraGalacticIntensity(maxf, zs, Ms, Es, Ned, Mstar, M, comptonTreatment = compton, includeICS=False, includePositron=True)
#    rhoEarth_eV = rhoEarth_GeVcm3 / GeVpereV * (hbar*c/mpercm)**3 #eV^4
#    rNFW_eV = rNFW / kpcPerMpc / eVMpc #eV
#    rEarth_eV = rEarth / kpcPerMpc / eVMpc #eV
#    
#    galI = galacticIntensity(Es, maxf*Ms[-1]/M, Ms[-1], Ned, Mstar, rhoEarth_eV,rNFW_eV,rEarth_eV, gNFW, includePositron = True)
#    plt.loglog(Es,galI,'r',label='Gal')#,linewidth=1)
#    plt.loglog(Es,Is,'g',label='EBL')#,linewidth=1)
#    Is += galI
#    np.savetxt('N %d M %.1e g spectrum.txt'%(Ned,M_g),np.array([Es, Is]).T) 
#    plt.loglog(Es,Is,'k',label='Theory')#,linewidth=1)
##    #maxf=1
##    print('%e'%maxf)

##    egretData = np.loadtxt('GammaData/EGRET-2004.csv',delimiter=',',skiprows=1)

##    #EmidEgret = egretData[:,0]
##    EwidthEgret = (egretData[:,4]-egretData[:,3])
##    EmidEgret = (egretData[:,4]+egretData[:,3])/2
##    IEgret = egretData[:,1] #* EwidthEgret
##    dIEgret = egretData[:,2]#* EwidthEgret

##    chandraData = np.loadtxt('GammaData/Chandra_XMM-2004.csv',delimiter=',',skiprows=1)

##    #EmidChandra = chandraData[:,0]
##    EwidthChandra = (chandraData[:,4]-chandraData[:,3])
##    EmidChandra = (chandraData[:,4]+chandraData[:,3])/2
##    IChandra = chandraData[:,1] #* EwidthChandra
##    dIChandra = chandraData[:,2]#* EwidthChandra

##    swiftData = np.loadtxt('GammaData/SWIFT_BAT-2008.csv',delimiter=',',skiprows=1)

##    #EmidSwift = swiftData[:,0]
##    EwidthSwift = (swiftData[:,4]-swiftData[:,3])
##    EmidSwift = (swiftData[:,4]+swiftData[:,3])/2
##    ISwift = swiftData[:,1] #* EwidthSwift
##    dISwift = swiftData[:,2]#* EwidthSwift

##    nagoyaData = np.loadtxt('GammaData/Nagoya-1975.csv',delimiter=',',skiprows=1)

##    #EmidNagoya = nagoyaData[:,0]
##    EwidthNagoya = (nagoyaData[:,4]-nagoyaData[:,3])
##    EmidNagoya = (nagoyaData[:,4]+nagoyaData[:,3])/2
##    INagoya = nagoyaData[:,1] #* EwidthNagoya
##    dINagoya = nagoyaData[:,2]#* EwidthNagoya

##    comptelData = np.loadtxt('GammaData/COMPTEL-2000.csv',delimiter=',',skiprows=1)

##    #EmidComptel = comptelData[:,0]
##    EwidthComptel = (comptelData[:,4]-comptelData[:,3])
##    EmidComptel = (comptelData[:,4]+comptelData[:,3])/2
##    IComptel = comptelData[:,1] #*EwidthComptel
##    dIComptel = comptelData[:,2]#*EwidthComptel

##    fermiData = np.genfromtxt('GammaData/FERMI_LAT-2015.txt',skip_header=61,skip_footer=52)

##    EmidFermiA = (fermiData[:,2] + fermiData[:,1])/2
##    EwidthFermiA = (fermiData[:,2] - fermiData[:,1])
##    IFermiA = fermiData[:,3]/EwidthFermiA
##    dIFermiA = np.array([np.sqrt(fermiData[:,4]**2 + fermiData[:,6]**2), np.sqrt(fermiData[:,5]**2 + fermiData[:,7]**2)])/EwidthFermiA 

##    plt.errorbar(1e6*EmidChandra, 1e3*IChandra, 1e3*dIChandra, 1e6*EwidthChandra/2, fmt='none')
##    plt.errorbar(1e6*EmidSwift, 1e3*ISwift, 1e3*dISwift, 1e6*EwidthSwift/2, fmt='none')
##    plt.errorbar(1e6*EmidNagoya, 1e3*INagoya, 1e3*dINagoya, 1e6*EwidthNagoya/2, fmt='none')
##    plt.errorbar(1e6*EmidComptel, 1e3*IComptel, 1e3*dIComptel,1e6* EwidthComptel/2, fmt='none')
##    plt.errorbar(1e6*EmidEgret, 1e3*IEgret, 1e3*dIEgret, 1e6*EwidthEgret/2, fmt='none')
##    plt.errorbar(1e6*EmidFermiA, 1e3*IFermiA, 1e3*dIFermiA, 1e6*EwidthFermiA/2, fmt='none')#, capsize=2,linewidth=1)

#    constraints = maxIntensity(sigmaBound = 0, plot=True)
#    #fluxErrors = (maxIntensity(sigmaBound = 1)-constraints)[:,1]
#    
#   # binWidth = constraints[:,3] - constraints[:,2]
#    #plt.errorbar(constraints[:,0], constraints[:,1], fluxErrors, binWidth/2, fmt='none')

##    plt.yscale('log')
##    plt.xscale('log')
#    plt.yscale('linear')
#    plt.xscale('linear')
#    #plt.tight_layout()
#    #plt.xlim([4e5, 7e5])
#    #plt.ylim([1e1,1e4])
#    #plt.ylim([np.min(constraints[:,1])/10, np.max(constraints[:,1])*10])
#    #plt.ylim([np.min(1e3*IFermiA)/10, np.max(galI)*10])
#    #plt.legend(['Chandra', 'Swift', 'Nagoya', 'COMPTEL', 'EGRET', 'FERMI-LAT'])
#    plt.ylabel('Flux (photon/cm$^2$/s/sr/eV)',fontsize=16)
#    plt.xlabel('Energy (eV)',fontsize=16)
#    #plt.legend(ncol=2,fontsize=11)
#    #plt.ylim([1e0,1e2])
#    plt.ylim([0,40])
#    plt.xlim([5e5,6e5])
#    #plt.title('$N_{ED}=5$, $M=10^{16}$g, $M_*=10$TeV',fontsize=14)
#    #plt.title('N = %d M = %.1e g'%(Ned,M_g))
#    #plt.savefig('N5 M1e16g spectrum.pdf')
#    plt.show()
#    
#    exit()

    Mstar = 10e12
    Mstar_GeV = 10e3

#    NmED = 1 #Number of masses when using extra dimensinos 
    #NmED = 60
    NmED = 5
#    NmED = 27
    usePositrons = True #whether this scan should include the effect of positronium
    useICS = False #whether this scan should include the effect of ICS
    useGalaxy = True #whether this scan should use contribution of MW Halo
    useAjello = True #whether this scan should use Marco Ajello's Data

    comptonMethod = 'Ignore'

    #Neds = [2, 3, 4, 5, 6]
    #logMmins = [4, 7, 9, 10, 12]
    #logMmaxs = [12, 16, 18, 18, 18]
    Neds = [0]
    logMmins = [17]
    logMmaxs = [np.log10(1.8e17)]
#    Neds = [4, 5, 6]
#    logMmins = [9, 10, 12]
#    logMmaxs = [18, 17, 17]

    #Neds = [2]
    #logMmins = [5]
    #logMmaxs = [5]
    for i, Ned in enumerate(Neds):
        Nm = NmED
        print('%d extra dimensions'%Ned)
        Mgs = np.logspace(logMmins[i], logMmaxs[i], Nm) #masses in grams

        initfdms = np.zeros(Nm)
        bps = np.zeros(Nm)
        rhoDM = Omdm * rhoc0_gcm3 *(hbar*c/mpercm)**3 * eVperg
        for i, M_g in enumerate(Mgs):
            #print(M_g)
            M = M_g * eVperg
            initfdms[i] = maxfdm(M, Ned, Mstar, includeICSSpec = useICS, includeGalactic =useGalaxy, includePositronSpec=usePositrons, useAjello=useAjello, comptonTreatment = comptonMethod)

            print("M = %e g   Max fdm_i = %g"%(M_g, initfdms[i]))

#        plt.legend(['$M = %.3e$'%m for m in Mgs])
#        plt.xlim([100,1100])
#        plt.show()

        fileName = '%d Ned Constraint'%Ned
        if useAjello:
            fileName += ' AjelloData'
        else:
            fileName += ' AviData'
        if useICS:
            fileName += ' with ICS'
        else:
            fileName += ' without ICS'
        if useGalaxy:
            fileName += ' with galaxy'
        else:
            fileName += ' without galaxy'
        if usePositrons:
            fileName += ' with positrons'
        else:
            fileName += ' without positrons'
        fileName += ' Compton'
        fileName += comptonMethod
        fileName += '.csv'
        np.savetxt(fileName,np.array([Mgs,initfdms]).T,delimiter=',')
#        plt.loglog(Mgs,initfdms,'.')
#        plt.show()
    
#    plt.xlabel('$M_{PBH}$ (g)')
    #plt.ylabel('$f_{dm,i}$')
    #plt.legend(['$N_{ED} = %d$'%n for n in Neds])
    #plt.xlim([1e4,1e18])
    #plt.ylim([1e-10,1e-0])
    #plt.xlim([9e13,5e14])
    #plt.ylim([2e-5,9e-4])
    #plt.ylim([1e-28,1e-12])
    #plt.xlim([1e9,5e18])
    #plt.ylim([1e-10, 1e-0])
    #plt.savefig("constraints.png") 
#    plt.show()

#Ned = 0
##Mis = np.logspace(np.log10(4.95e10), 17.5, 250)
#Mis = 4.7999225960e+14 + np.logspace(5,18,150)
#Mfs = np.zeros_like(Mis)
#t0 = 4.354e17
#for i, Mi in enumerate(Mis):
#    Mfs[i] = calcMoft(np.array([t0]), Mi*eVperg, Ned, 10e12, odeMethod = 'BDF')[0]/eVperg #Black hole masses in eV
#    print('%.10e\t%.2e'%(Mi,Mfs[i]))
#output=np.array([Mis,Mfs]).T
#np.savetxt('MbhInitialtoTodayMapping_%d.csv'%Ned,output,delimiter=',')
#Mbh = 4.9e14*eVperg
#t0 = 4.354e17
#lt = lifetime(Mbh, Ned)
#print(lt)
#while lt <  t0:
#    Mbh += 0.001e14*eVperg
#    lt = lifetime(Mbh, Ned)
#    print('%.3e\t%.3e'%(Mbh/eVperg, lt))
#plt.show()


#main()

#T_eV = 30e6
#M = 1 / (8*pi*G_eV*T_eV)
n = 2
M = 1e22

T_GeV = get_temperature_from_mass(M,2,10e3)
print(T_GeV*1e9)
#E = np.logspace(3,9.5,100)
#dNdEdt = dNgamdotdE(M, E, 0, Mstar=10e12, primaryOnly=False) / hbar 
#plt.loglog(1e-9*E, 1e-9*np.power(E,2)*dNdEdt, label='Us')

##carr = np.loadtxt('CarrSpec30MeV.csv',delimiter=',')
##plt.loglog(carr[:,0], carr[:,1], label='Carr')

#chen = np.loadtxt('ChenSpec1e15g.csv',delimiter=',')
#plt.loglog(chen[:,0], chen[:,1], label='Chen (BlackHawk 2.0)')

#plt.legend()
#plt.title('M = 1e15 g')
#plt.xlabel('E (GeV)')
#plt.ylabel('E$^2$ dN/dEdt [GeV/s]')
##plt.xlim([2e7, 5e8])
#plt.ylim([1e13, 1e19])
#plt.show()
