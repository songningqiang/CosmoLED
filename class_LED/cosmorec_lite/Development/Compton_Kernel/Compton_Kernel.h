//====================================================================================================================
// Author Jens Chluba March 2011
//====================================================================================================================

//====================================================================================================================
// Purpose: Implementation of electron scattering kernel functions following
// Sazonov & Sunyaev, 2000, Apj 543, p. 28-55.
//====================================================================================================================

//--------------------------------------------------------------------------------------------------------------------
// Th_e == k Te / me c^2
// delta == [nu'-nu]/[nu'+nu]/sqrt(2 Th_e)
//--------------------------------------------------------------------------------------------------------------------

#ifndef COMPTON_KERNEL_H
#define COMPTON_KERNEL_H

#include "physical_consts.h"

//====================================================================================================================
//
// conversion nu' <--> delta
//
//====================================================================================================================
inline double Kernel_delta(double nu, double nup, double Th_e){ return (nup-nu)/(nup+nu)/sqrt(2.0*Th_e); }
inline double Kernel_deltaxe(double nu, double nup, double Tm){ return const_h_kb*(nup-nu)/Tm; }

//====================================================================================================================
//
// lowest order kernel; no recoil, no Doppler boosting;
//
//====================================================================================================================

double P0_Kernel(double nu, double nup, double Th_e);
double P0_Kernel_spline(double nu, double nup, double Th_e);
double norm_P0(double nu, double Th_e);

double P0_Kernel_AliHaimoud(double nu, double nup, double Th_e);

//====================================================================================================================
//
// 'Kompaneets' kernel; Doppler broadening, Doppler boosting, and electron recoil terms are included
//
//====================================================================================================================

double PK_Kernel(double nu, double nup, double Th_e);
double PK_Kernel_spline(double nu, double nup, double Th_e);
double norm_PK(double nu, double Th_e);

//====================================================================================================================
//
// 'full' kernel according to Eq. (19) of Sazonov & Sunyaev, 2000
//
//====================================================================================================================

double P_Compton(double nu, double nup, double Th_e);

//====================================================================================================================
//====================================================================================================================

#endif
