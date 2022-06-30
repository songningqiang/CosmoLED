
double od_dtdl_photion( double z, double eV );
double od_dtdl_Compton( double z, double eV );
double od_dlnEdl_Compton( double z, double eV );
double od_dtdl_pp_atom( double z, double eV );
double od_dtdl_pp_ions( double z, double eV );
double od_dtdl_2photon( double z, double eV );
double od_dtdl_pp_cmb1( double z, double eV );
double od_dtdl_pp_cmb2( double z, double eV );

double od_dldlz( double z );

double od_dtdz(double z, double eV);
void od_dtdzVec(double z, double* eVs, double* dtdzs, int NE);
void od_dEdzComptonVec(double z, double* eVs, double* dEdzs,int NE);
