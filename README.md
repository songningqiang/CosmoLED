# Cosmo code for Large Extra Dimension (LED) black holes

This is a code to compute Hawking evaporation from black holes and set constraints on the fraction of black holes. Although the code is designed for large extra dimension black holes, 4D black holes can also be studied with a proper setup. The code also improves over the previous code in various aspects. Please see [https://arxiv.org/pdf/2201.11761.pdf](https://arxiv.org/pdf/2201.11761.pdf) for details.

## Instructions for class_LED

The code is developed based on [ExoCLASS](https://github.com/lesgourg/class_public/tree/ExoCLASS). ExoCLASS and the DarkAges module are still usable in their vanilla way. Additional features are introduced with the DarkAges_LED module and the C codes in class_LED to compute the evolution and energy deposition functions from LED black holes. Greybody factors, lepton and meson decay tables are available in the class_LED/DarkAges/data folder.

DarkAges_LED requirements:
- numpy
- scipy
- PyYAML
- dill
- future

DarkAges_LED can be used separately to obtain the energy deposition functions. For example, under class_LED folder, one can use
```
python DarkAgesModule/bin/DarkAges_LED --hist=evaporating_PBH --mass=1e+15 --dims=6 --mpl=1e4 --outfile=testfz/f_z_PBH_6d_1e4_1e15.0g_AllChannels.dat
```
where the different parameters are:
- mass: the mass of black hole in grams.
- dims: number of extra dimensions, can choose from 0 to 6.
- mpl: bulk Planck scale in GeV. When --dims=0, set mpl=Mpl, mpl or 1.22e19 to deal with traditional 4D black holes.
- outfile: the file to write in energy deposition functions.

For a list of options available, one can try
```
python DarkAgesModule/bin/DarkAges_LED --help
```
Note that errors may occur when using different python versions (python 2 vs python3) subsequently. Clear all obj files if this happens.

An example of using the energy deposition function file in class_LED is shown in testBH_efffile.ini. Basically one needs to set
```
on the spot = no
energy_deposition_function = from_file
energy deposition function file = name_of_function_file
```
along with the properties of the black holes.

Alternatively, one could run class_LED directly without generating energy deposition function file. An example is test_LEDBH.ini with the following lines
```
energy_deposition_function = DarkAges_LED
DarkAges_mode = built_in
```

It is also possible to set PBH_fraction on log scale. To enable this feature, one can set log10_PBH_fraction = yes in the ini file and then set PBH_fraction = log10(PBH_fraction). By default log10_PBH_fraction = no.

To use class _LED with MontePython, it is highly recommended to produce the energy deposition file prior to running MCMC, as calling the DarkAges_LED module each time is time consuming. Make sure the path of the file is correct. An example parameter file for MontePython using Planck 2018 data can be found as LED_6d_1e4_1e15g.param.

## Instructions for IsotropicLight

The code for setting constraints on LED black holes using isotropic background light can be found in the IsotropicLight folder. The relevant code for the calculation is in the file isotropicLight.py which is dependent on the following python packages:
- numpy
- matplotlib
- scipy
- ctypes

The isotropic light constraints relies on photon optical depth code written by Katherine Mack and Daniel Wesley with explanations found in [arxiv:0805.1531](https://arxiv.org/abs/0805.1531). Before running the isotropicLight python code, the photon optical depth code must be compiled. This can be done by running Make all in the IsotropicLight folder.

The isotropicLight.py code can be run by calling
```
python3 isotropicLight.py
```
This will allow you to choose either to scan over all black hole masses and number of extra dimensions or find the constraint for a single value. The constraint calculation is only implemented currently for a bulk Planck scale of 10 TeV. Constraints for traditional 4D primordial black holes can be obtained by choosing 0 for the number of extra-dimensions. In the case of 4D black holes, the bulk Planck scale is not used because it is assumed to be the observed 4D Planck scale. When running isotropicLight.py in its current form you will be prompted to choose an approximation for the Compton scattering calculation. An explanation of each calculation method can be found in Appendix D.2 of [the associated paper](https://arxiv.org/pdf/2201.11761.pdf).

To see an example for how to use the code to directly calculate constraints see the functions mainIndividual and mainScan. A description of all the functions provided in isotropicLight.py can be found in documentation.txt.

Calculation of particle spectra is reliant on interpolating tables from [A Poor Particle Physicist Cookbook for Dark Matter Indirect Detection (PPPC4DMID)](https://arxiv.org/abs/1012.4515). Those tables along with tables used to interpolate the black hole greybody factors, and some additional helpful functions for black hole evolution can be found in the folder PPPC4DMID.

Setting constraints requires experimental data. All x-ray and gamma ray telescope data is in the folder AjelloData. This data contains observations from a large number of telescopes and was provided courtesy of Marco Ajello. Instructions for how to update the dataset used are also found in the folder AjelloData.

## Using the code:

Feel free to use, modify or distribute the code. If you use the code in your publication, please cite the paper [https://arxiv.org/pdf/2201.11761.pdf](https://arxiv.org/pdf/2201.11761.pdf)

Any use of the IsotropicLight code should also cite "M. Ajello et al., “Cosmic X-ray background and Earth albedo Spectra with Swift/BAT,” Astrophys. J. 689, 666 (2008), [arXiv:0808.3377](https://arxiv.org/abs/0808.3377) [astro-ph]." as the source of the collected X-ray and gamma ray background data provided by Marco Ajello. 

Additionally, use of the IsotropicLight code should cite the PPPC4DMID paper: [arxiv:1812.4515](https://arxiv.org/abs/1012.4515) for use of the secondary particle spectra.
