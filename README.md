# Cosmo code for Large Extra Dimension (LED) black holes

## Instructions for class_LED

The code is developed based on [ExoCLASS](https://github.com/lesgourg/class_public/tree/ExoCLASS). ExoCLASS and the DarkAges module are still usable in their vanilla way. Additional features are introduced with the DarkAges_LED module and the C codes in class_LED to compute the evolution and contribution from LED black holes. Greybody factors, lepton and meson decay tables are available in the class_LED/DarkAges/data folder.

DarkAges_LED requirements:

numpy
scipy
PyYAML
dill
future

DarkAges_LED can be used separately to obtain the energy deposition functions. For example, under class_LED folder, one can do

python DarkAgesModule/bin/DarkAges_LED --hist=evaporating_PBH --mass=1e+15 --dims=6 --mpl=1e4 --outfile=testfz/f_z_PBH_6d_1e4_1e15.0g_AllChannels.dat

mass: the mass of black hole in gram.
dims: number of extra dimensions, can choose from 0 to 6.
mpl: bulk Planck scale in GeV. When --dims=0, set mpl=Mpl, mpl or 1.22e19 to deal with traditional 4d black holes.
outfile: the file to write in energy deposition functions.

for a list of options available, one can try

python DarkAgesModule/bin/DarkAges_LED --help

Note that errors may occur when using different python versions (python 2 vs python3) subsequently. Clear all obj files if this happens.

An example of using the energy deposition function file in class_LED is shown in testBH_efffile.ini. Basically one needs to set
on the spot = no
energy_deposition_function = from_file
energy deposition function file = name_of_function_file
along with the properties of the black holes.

Alternatively, one could run class_LED directly without generating energy deposition function file. An example is test_LEDBH.ini with the following lines

energy_deposition_function = DarkAges_LED
DarkAges_mode = built_in

input.c has been modified to allow PBH_fraction on log scale. To enable this feature, one can set log10_PBH_fraction = yes in the ini file and then set PBH_fraction = log10(PBH_fraction). By default log10_PBH_fraction = no.

To use class _LED with MontePython, it is highly recommended to produce the energy deposition file prior to running MCMC, as calling the DarkAges_LED module each time is time consuming. Make sure the path of the file is correct. An example parameter file for MontePython using Planck 2018 data can be found as LED_6d_1e4_1e15g.param.

## Instructions for IsotropicLight

## Using the code:

Feel free to use, modify or distribute the code. If you use the code in your publication, please cite the paper [https://arxiv.org/pdf/2201.11761.pdf](https://arxiv.org/pdf/2201.11761.pdf)
