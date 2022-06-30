def H
	Returns the expansion rate of the universe at redshift z in eV

    Parameters
    ----------
    z : float or numpy array
        Redshift(s) to have the expansion rate evaluated at

    Returns
    -------
    same type as z
        Expansion rate in eV
    

def NtodNdE
	Converts a total spectrum N to a differential spectrum dN/dE

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
    

def PBH_dMdt
	Returns the differential mass loss rate of an primordial black hole with
	a mass of :code:`PBH_mass*scale` gramm.

	This method calculates the differential mass loss rate at a given (rescaled) mass
	of the black hole and at a given time, given by

	.. math::

	   \frac{\mathrm{d}M\,\mathrm{[g]}}{\mathrm{d}t} = - 5.34\cdot 10^{25}
	   \cdot \mathcal{F}(M) \cdot \left(\frac{1}{M\,\mathrm{[g]}}\right)^2
	   \,\frac{\mathrm{1}}{\mathrm{s}}

	.. note::

	   Even if this method is not using the variable :code:`time`, it is needed for the
	   ODE-solver within the :meth:`PBH_mass_at_t <DarkAges.evaporator.PBH_mass_at_t>`
	   for the correct interpretation of the differential equation for the mass of
	   the black hole.

	Parameters
	----------
	PBH_mass : :obj:`float`
		(Rescaled) mass of the black hole in units of :math:`\mathrm{scale} \cdot \mathrm{g}`
	time : :obj:`float`
		Time in units of seconds. (Not used, but needed for the use with an ODE-solver)
	scale : :obj:`float`, *optional*
		For a better numerical performance the differential equation can be expressed in
		terms of a different scale than :math:`\mathrm{g}`. For example the choice
		:code:`scale = 1e10` returns the differential mass loss rate in units of
		:math:`10^{10}\mathrm{g}`. This parameter is optional. If not given, :code:`scale = 1`
		is assumed.

	Returns
	-------
	:obj:`float`
		Differential mass loss rate in units of :math:`\frac{\mathrm{g}}{\mathrm{s}}`
	

def PBH_primary_spectrum
	Returns the double differential spectrum
	:math:`\frac{\mathrm{d}^2 N}{\mathrm{d}E \mathrm{d}t}` of particles with
	a given :code:`spin` and kinetic :code:`energy` for an evaporating black hole
	of mass :code:`PBH_mass`. Antiparticle not included.

	Parameters
	----------
	energy : :obj:`float`
		Kinetic energy of the particle (*in units of* :math:`\mathrm{GeV}`)
		produced by the evaporating black hole.
	PBH_mass : :obj:`float`
		Current mass of the black hole (*in units of* :math:`\mathrm{g}`)
	spin : :obj:`float`
		Spin of the particle produced by the evaporating black hole (Needs
		to be a multiple of :math:`\frac{1}{2}`, i.e. :code:`2 * spin` is assumed
		to have integer value)

	Returns
	-------
	:obj:`float`
		Value of :math:`\frac{\mathrm{d}^2 N}{\mathrm{d}E \mathrm{d}t}`

def Thawk4D
	Calculated the Hawking temperature of a 4D blackhole

    Parameters
    ----------
    Mpbh : float or numpy array
        Mass of the blackhole in eV

    Returns
    -------
    same type as Mpbh
        Temperature of blackhole in eV. Returns -1 if the mass is zero or negative
    

def bhDensity
	The number density of black holes today if black holes of mass M0 (eV)
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

def calcDimensionRadius
	Returns the size of the extra dimensions

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
    

def calcMoft
	Calculate black hole mass for a function of time (seconds) for a given initial mass (eV)

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

def dMdt4D
	\frac{dM}{dt}, change in 4D black hole mass in eV/s

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
    

def dMdtED
	\frac{dM}{dt}, change in black hole mass in eV/s in case of large extra dimensions

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
    

def dNdEtoN
	Converts a differential spectrum dN/dE to a total spectrum N(E)

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
    

def dNelecdotdE
	\frac{d^2N_\electron}{dEdt}, the differential number
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
    

def dNelecdotdESingleM
	\frac{dN_\electron}{dEdt}, the differential number
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
    

def dNelecdt
	Calculates the rate of production of electrons and positrons 
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
    

def dNgamdotdE
	\frac{dN_\gamma}{dEdt}, the differential number
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
    

def dNgamdotdECompton
	Calculates the instataneous rate of change of a flux of photons due to
    Compton scattering at reshift z. This solves the equation:
    $\frac{dN}{dEdt} = -n_e(z)\sigma(E)\frac{dN}{dE}(E) + \frac{n_e m_e}{E^2}\int dE' \frac{d\sigma(E')}{d\cos (\theta)} \frac{dN}{dE'}(E')$
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
    

def dNgamdotdESingleM
	\frac{dN_\gamma}{dEdt}, the differential number
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
    

def dNgamdotdEVaryingM
	\frac{dN_\gamma}{dEdt}, the differential number
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
    

def dNgamdotdEpri4D
	\frac{dN_\gamma}{dEdt}, the differential number
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
    

def dTau
	calculates the differential photon absorption $\frac{d\tau}{dz}\delta z$
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
    

def dsigmadcosthCompton
	Differential Compton cross section in the rest frame of the electron for an incoming
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
    

def extraGalacticIntensity
	Calculates the Intensity of photons at energy E (eV) today
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
                          $abs(n_e(z) \sigma(E) \Delta t) << 1$. If that is larger than 1, 
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
    

def galacticIntensity
	Calculates the isotropic Intensity of photons at energy E (eV) today
     in $cm^{-2} s^{-1} sr^{-1}GeV^{-1}$ from black hole of mass M
    within the Milky Way halo.

    This assumes that the dark matter halo obeys the the gNFW profile of
    $\rho(r) = \rho_{earth}(\frac{r_{earth}}{r})^g (\frac{r_s + r_{earth}}{r_s + r})^{3-g}$.
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
        Radius parameter, $\r_s$, in the NFW profile in eV^-1
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

def icsTotaldNgamdE
	Calcultes the differential photon spectrum $\frac{dN_\gamma}{dE}$, 
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
    

def icsdNgamdE
	Calcultes the differential photon spectrum $\frac{dN_\gamma}{dE}$, 
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
    
def isMacroscopic
	Returns whether BH is larger than the size of extra dimensions

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
    

def mainIndividual
	
    Main function used for determing max_fdm for a single black hole mass
    

def mainScan
	
    Main function used for determing max_fdm for a scan over varying black hole mass for n=2-6 extra dimensions
    

def maxIntensity
	Outputs array with measured values of photon intensity.

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
    

def maxfdm
	Determines the constraint on fdm, the fraction of
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
    
