/*
    Implementation of absorption formulae found in:
    Zdziarski, Svensson.  ApJ _344_ (1989) 551.
    
    Currently a simple implementation, a more complete one would
    take account of the ionization fraction, etc., at various
    redshifts.  As-is, the caller will have to be aware of some of
    the subleties.
    
    To-Do:
    
     1. The photoionization cross-section is a fit for E > 250 eV.  There are
        fits available to 25 eV in the ZS paper, and probably elsewhere in the
        literature, that could be added in here to extend the range.  Right now
        the od_dtdl_photoion() function just returns zero for energies outside
        of the range.
     
     2. A similar situation exists for the pair production off atoms, and so
        the od_dtdl_pp_atom() function returns zero for inputs outside of the
        stated range.
        
     3. Similarly, in od_dtdl_pp_ions() the formulae are only valid for epsilon
        much greater than one, though exact expressions appear to be avail-
        able in the literature (see the paper for more information)
     
     4. In od_dtdl_pp_cmb1(), the formulae is really a patch of two different
        approximate formulae. (Exact expressions again appear to be available
        in the literatre).  The trouble will happen when epsilon*theta is of
        order unity.  (these are the photon energy and CMB temperature, resp.,
        in units of the electron mass).
     
     5. In od_dtdl_pp_cmb2() the expression appears to depend on the energy in
        the CM frame, as well as potentially only being valid for a certain 
        range of the epsilon/theta parameter space.  This needs to be sorted out
        (right now the code just returns the value of the given expression)
    
    Changes log:
        
    2007 05 29  DHW  Created file.
    2007.06.28  KJM  Have added function to find dominant absorption mechanism.
    2007.07.16  DHW  Removed this function (reasons below).
           .21       Removed local #defines for constants
                     Extended range of od_dtdl_photion to 25 eV.
        .09.07       Added a parens in #define
                     Put a low-energy cutoff in double e/p pp off CMB
                     
*/

#include "lib.h"

/* I have modified these to accept in units of h instead of H0 from ZS       */
#define TAU0_PHOTION ((2.0e-10)*(TODAY_OB/0.1)*(TODAY_h/.5)) /* 3.4 in ZS    */
#define TAU0_COMPTON ((3.03e-3)*(TODAY_OB/0.1)*(TODAY_h/.5)) /* 4.2 in ZS    */
#define TAU0_PP_ATOM ((1.4e-5)*(TODAY_OB/0.1)*(TODAY_h/.5))  /* 5.3 in ZS    */
#define TAU0_PP_IONS ((1.75e-5)*(TODAY_OB/0.1)*(TODAY_h/.5)) /* 5.9 in ZS    */
#define TAU0_2PHOTON ((1.83e-27)*(TODAY_OB/0.1)*(TODAY_h/.5)) /* 6.1 in ZS   */
#define TAU0_PP_CMB1 ((3.83e5)*CUBE(T_CMB0/2.7)/(TODAY_h/.5)) /* 7.2 in ZS   */
#define TAU0_PP_CMB2 ((47.5)*CUBE(T_CMB0/2.7)/(TODAY_h/.5))   /* 7.10 in ZS   */

/* Handy shorthand since many functions define eps and use it */
#define LOAD_EPS    double eps; eps=eV/E_RM_eV;

/* For use by tau max finder */
#define max(A,B) ( (A) > (B) ? (A):(B)) 

#include <math.h>
#include <stdio.h>


/*
    This returns the Hubble length today over the Hubble length at z.  The 
    reason is that the formulae given by ZS do dimensionless scattering 
    probabilities per today's Hubble length.  If one wishes to get the corres-
    ponding value for far in the past, multiply by the output of this function.
*/

double od_dldlz( double z )
{
    return sqrt( TODAY_OL + TODAY_OM*CUBE(1.0+z) + \
                 TODAY_OR*POW4(1.0+z) );
}

/*
    The dimensionless scattering probability for photoionization from neutral
    gas at photon energies E > 250 eV.  See ZS equation 3.3.  If the energy
    falls below this range, the function returns zero.
*/

double od_dtdl_photion( double z, double eV )
{
    if (eV < 25.)
        return 0.;
    else if (eV < 250.)
        return TAU0_PHOTION * CUBE(1.0+z) * pow( eV/E_RM_eV, -2.65 );

    return TAU0_PHOTION * CUBE(1.0+z) * pow( eV/E_RM_eV, -3.3 );
}


/*
    Compton scattering by cold matter.
    See ZS equation 4.1a
*/

double od_dtdl_Compton( double z, double eV )
{
    double i_eps, f;
    LOAD_EPS
    
    i_eps = 1.0/eps;
    
    f = (0.375*i_eps) * ( (1.0-2.0*i_eps-2.0*SQUARE(i_eps))*log(1.0+2.0*eps) \
                          + 4.0*i_eps \
                          + 2.0*eps*(1.0+eps)/SQUARE(1.0+2.0*eps) );
    
    return TAU0_COMPTON * CUBE(1.0+z) * f;
}


/*
    The logarithmic energy loss per Hubble time.
    ZS 4.8
*/

double od_dlnEdl_Compton( double z, double eV )
{
    LOAD_EPS
    double i_eps, g;
    
    i_eps = 1.0/eps;
    g = 0.375 * ( ((eps-3.0)*(eps+1.0)/POW4(eps)) * log(1.0 + 2.0*eps) + \
                  2.0*(3.0 + 17.0*eps + 31.0*SQUARE(eps) + 17.0*CUBE(eps) - \
                      10.0*POW4(eps)/3.0)/(CUBE( eps + 2.0*SQUARE(eps))));
    
    return TAU0_COMPTON*CUBE(1.0+z)*eps*g;
}


/*
    Pair production from atoms (H and He).
    ZS equation 5.3
    
    This assumes that epsilon > 6.
*/

double od_dtdl_pp_atom( double z, double eV )
{
    LOAD_EPS
    
    if (eps < 6.0)
        return 0.0;
    
    return TAU0_PP_ATOM * CUBE(1.0+z) * log( 513.0*eps/( eps + 825.0) );
}


/*
    Pair production from fully ionized matter.
    ZS equation 5.9.
    
    Assumes epsilon >> 1 (which I take to be >)

    2008.04.13  KJM  To prevent the optical depth going negative,
	setting to zero if eps < 6.7
*/

double od_dtdl_pp_ions( double z, double eV )
{
    LOAD_EPS
    
    if (eps < 6.7)
        return 0.0;
    
    return TAU0_PP_IONS * CUBE(1.0+z) * ( log(2.0*eps) - 2.5952380952380953 );
}


/*
    Photon-photon scattering
    ZS equation 6.1
*/

double od_dtdl_2photon( double z, double eV )
{
    LOAD_EPS
    
    return TAU0_2PHOTON * POW6(1.0+z) * CUBE(eps);
}


/*
    Photon-photon production of a single pair
    ZS equation 7.1
    
    Note: we are basically using two different approximations, that will
    be valid in different regimes
*/

double od_dtdl_pp_cmb1( double z, double eV )
{
    LOAD_EPS
    double th, eps_th;
    
    th = (1.0 + z) * T_CMB0 * BOLTZ_K_eV / E_RM_eV;
    eps_th = eps * th;
    
    if (eps_th > 1.0)
        return TAU0_PP_CMB1 * CUBE(1.0+z) * \
               6.5797362673929056 * log( 4.0 * eps_th - 2.1472)/eps_th;
    
    return TAU0_PP_CMB1 * CUBE(1.0+z) * \
           2.0 * sqrt( 3.1415926535897931/eps_th ) * \
           exp( -1.0/eps_th ) * (1.0 + 2.25*eps_th);
}


/*
    Photon-photon double pair production
    ZS 7.9
*/

double od_dtdl_pp_cmb2( double z, double eV )
{
    LOAD_EPS
    double th, eps_th;
    
    th = (1.0 + z) * T_CMB0 * BOLTZ_K_eV / E_RM_eV;
    eps_th = eps * th;
    
    if (eps_th > 1.)
        return TAU0_PP_CMB2 * CUBE(1.0+z);
    
    return 0.;
}


/* This function finds tau_eff / dt 

DHW: i commented this out because you can just add optical depths.
     also, od_dlnEdl_Compton() does not return an optical depth, but an
     energy loss.

double od_dtau_eff(double z, double eV)
{

    double dtau1,
         dtau;

    Find the dominant process and use this for the optical depth. 
    There may be a less clumsy way of doing this 

    dtau1 = od_dtdl_photion(z,eV);

    dtau1 = max(dtau1, od_dlnEdl_Compton(z,eV));

    dtau1 = max(dtau1, od_dtdl_pp_atom(z,eV));

    dtau1 = max(dtau1, od_dtdl_pp_ions(z,eV));

    dtau1 = max(dtau1, od_dtdl_2photon(z,eV));

    Return dtau/dt, units are s^-1 
    dtau = dtau1 * TODAY_H0;

    return dtau;

} */

double od_dtdz(double z, double eV){
	double dtdl = 0;
	if(z > 6)
		dtdl += od_dtdl_photion( z, eV );

	dtdl += od_dtdl_Compton( z, eV );
	dtdl += od_dtdl_pp_atom( z, eV );
	dtdl += od_dtdl_pp_ions( z, eV );
	dtdl += od_dtdl_2photon( z, eV );
	dtdl += od_dtdl_pp_cmb1( z, eV );
	dtdl += od_dtdl_pp_cmb2( z, eV );
	
	double dldz = od_dldlz( z )*(1+z);
	return dtdl/dldz;
}

void od_dtdzVec(double z, double* eVs, double* dtdzs,int NE){
	double dldz = od_dldlz( z )*(1+z);
	for(int i = 0; i < NE; i++){
		double eV = eVs[i];
		double dtdl = 0;
	    if(z > 6)
			dtdl += od_dtdl_photion( z, eV );

		dtdl += od_dtdl_Compton( z, eV );
		dtdl += od_dtdl_pp_atom( z, eV );
		dtdl += od_dtdl_pp_ions( z, eV );
		double T0 = 2.32e-4; //eV
		if(eV*T0*(1+z) < (511e3 * 511e3))
            dtdl += od_dtdl_2photon( z, eV );
		dtdl += od_dtdl_pp_cmb1( z, eV );
		dtdl += od_dtdl_pp_cmb2( z, eV );
		
		dtdzs[i] = dtdl/dldz;
	}
	return ;
}

void od_dEdzComptonVec(double z, double* eVs, double* dEdzs,int NE){
	double dldz = od_dldlz( z )*(1+z);
	for(int i = 0; i < NE; i++){
		double eV = eVs[i];
		double dlnEdl = od_dlnEdl_Compton( z, eV );
		
		dEdzs[i] = eV*dlnEdl/dldz;
	}
	return ;
}
