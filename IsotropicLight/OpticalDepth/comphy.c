/*
 |   comphy
 |
 |   Module to handle some physics stuff common to everyone
 |
 |   2007.07.02  DHW  Created file
 |
 */
 
 #include <math.h>
 #include "comphy.h"
 
 /*
 |   NAME: dtdz
 |   
 |   ARGS: z     Redshift
 |   
 |   RETS: dtdz at that redshift (as a POSITIVE NUMBER)
 |   
 |   DESC: dt / dz (z)
 |
 */

double dtdz( double z )
{    

    return 1.0/( TODAY_H0 * sqrt( TODAY_OM * POW5(1.0+z) + TODAY_OR * \
    POW6(1.0+ z) + TODAY_OL * SQUARE(1.0+z) ) );
}


/*
 |   NAME: Hubble
 |   
 |   ARGS: z     Redshift
 |   
 |   RETS: Hubble parameter at that redshift (in 1/s)
 |   
 |   Try speaking the function declaration three times fast.
 |
 */
 
 double Hubble( double z )
 {
    return TODAY_H0 * sqrt( TODAY_OM * CUBE(1.0+z) + TODAY_OR * \
                         POW4(1.0+ z) + TODAY_OL );
 }
