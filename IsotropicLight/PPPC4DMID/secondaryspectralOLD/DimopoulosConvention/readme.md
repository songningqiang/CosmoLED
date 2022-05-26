About the code:

1. Dimopoulos convention is implemented. See equation (3) of the overleaf draft.

2. Mstar = 10TeV is assumed for n>0, for n=0 Mstar=Mpl and we recover the 4d case.

3. Hadronization and decay of all particles are considered for primary particle energy higher than 5GeV by interpolating PPPC4DMID.

4. decay of pions, muons and taus are included for energy below 5GeV.

About the tables:

1. The data tables show dN/dlog10(E_kin)dt (in unit of GeV, or equivalently dNdlog10(x)dt where x=E_kin/Eprimary). To convert them to dN/dE_kindt you may divide them by (log(10)*E_kin). Then if you want them in unit of GeV^-1s^-1 you can divide them further by hbar. The particle energy E=E_kin+m.

2. The spectrum provided is for particle or antiparticle only, not both. A reasonable assumption is that there is no difference between particle and antiparticle spectrum.

3. One can assume proton=neutron=antiproton=antineutron if neutrons don't decay.

4. axes.txt has the axis information of the table, ie the rows correspond to particle kinetic energy and the columns correspond to different Hawking temperatures (all in log10 of GeV).