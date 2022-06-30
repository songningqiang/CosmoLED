
#define SQUARE(x)                ((x)*(x))
#define CUBE(x)                  ((x)*(x)*(x))
#define POW4(x)                  (SQUARE(x)*SQUARE(x))
#define POW5(x)                  (SQUARE(x)*CUBE(x))
#define POW6(x)                  (CUBE(x)*CUBE(x))

//#define TODAY_h             .704                        /* H0 in 100 km/s/Mpc */
#define TODAY_h             0.6736                       /* H0 in 100 km/s/Mpc */
#define TODAY_H0            (2.26854354e-18*(TODAY_h/.7)) /* Hz, for h=.7     */
//#define TODAY_OM            0.266
#define TODAY_OM            0.3153
#define Z_EQ                (1.0/(4.1707e-5/(TODAY_OM*SQUARE(TODAY_h)))-1.0)
                            /* z of matter-radiation equality                */
#define TODAY_OR            (TODAY_OM/(1.+Z_EQ))    /* use z_eq              */
#define TODAY_OL            (1.0-TODAY_OM-TODAY_OR) /* assume flat           */
//#define TODAY_OB            0.04411                 /* Omega_baryon today    */
#define TODAY_OB            0.0493                 /* Omega_baryon today    */
#define M_PROTON	    1.67262158e-27  /* Proton mass, in kg */
#define N_E0        ((2.46e-7)*(TODAY_OB/0.1)*SQUARE(TODAY_h/.5))
                            /* Current number density of electrons, (2.4) in 
                               the Zdziarski and Svensson reference (units?) */
#define E_RM_eV     510998.903    /* Electron rest mass, in eV               */

#define T_CMB0      2.728   /* CMB temperature, in Kelvin                    */

#define LN_10                    2.3025850929940459
#define PI                       3.14159265359
#define SOL                      2.99792458e8   /* c (m/s) */
#define SPEED_OF_LIGHT           SOL     /* m/s */
#define SOL2                     8.98755179e16  /* c^2 (m/s)^2 */
#define G_N                      6.6742000e-11 /* Newton's G kg m s units */
#define SB                       1.56055482e84  
        /* Stefan-Boltzmann constant for temperature in energy units (eg, emitted power (Watts) = SB*(kT)^4 Assumes two photon polarizations   */
#define EV_PER_JOULE             6.24150974e18  /* eV per Joule              */
#define EV_PER_KG              5.60958921e35  /* eV per kg                 */  

#define BOLTZMANN_K              1.3806505e-23  /* J/K */
#define BOLTZ_K_eV  8.6173423e-5  /* Boltzmann constant in eV/K              */
#define PLANCK_H        6.6260693000000002e-34  /*  Js                       */
#define HBAR            1.0545716823644548e-34  /*  Js                       */
#define RAD_DENS        3.78288454594e-16       /* pi^2 k4 / (30 hbar^3 c^3)
                                                   J / m^3 Hz 1 polarization!*/
#define THOMSON		6.65e-29		/* Thomson cross section in m^2 */
#define KG_PER_eV                1.78266173e-36
#define PLANCK_MASS     2.17586050198e-08       /*  kg                       */
#define PLANCK_TIME     5.39120570822e-44       /*  s                        */
#define PLANCK_ENERGY   1956096163.04           /*  J                        */

/* Number density of protons today in 1/m^3 */
#define PROT_DENS   ((7./8.)*3.0*SQUARE(TODAY_H0)*TODAY_OB/(8.0*PI*G_N*M_PROTON))
/* Number density of baryons today in 1/m^3 */
#define BARYON_DENS   (3.0*SQUARE(TODAY_H0)*TODAY_OB/(8.0*PI*G_N*M_PROTON))
#define EMIT_CONSTANT   1.71932504e37           /* Dimensionful constant for 
                                                   BH evap: hbar c^6/G^2, units
                                                   are kg^3 m^2/s^3          */
#define EMIT_M_CONS     (EMIT_CONSTANT/SQUARE(SOL))  /* Mass per time * mass^2*/
#define S_PER_YEAR      3.1556926e7
#define S_PER_GYR       3.1556926e16 

double dtdz( double z );
double Hubble( double z );
