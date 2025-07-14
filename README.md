![alt text](https://github.com/MDamiano/ExoReL/blob/main/ExoReL_logo.png?raw=true)

# ExoReL<sup>R</sup>

Version 2.4.2

Includes:
* A generation of reflected light spectra routine;
* A retrieval routine based on nested sampling (i.e. MultiNest).

## Authors
* [Mario Damiano](https://mdamiano.github.io/) (Jet Propulsion Laboratory, California Institute of Technology)
* [Renyu Hu](https://renyuplanet.github.io/) (Jet Propulsion Laboratory, California Institute of Technology)

## Collaborators
* Armen Tokadjian (Jet Propulsion Laboratory, California Institute of Technology)
* Zachary Burr (ETH Zurich, Jet Propulsion Laboratory, California Institute of Technology)

## Installation:
Install python packages dependency:

`pip install numpy scipy astropy scikit-bio matplotlib spectres`

Download the .zip file from Github, unzip it and place it in a folder at your preference. 
Therefore, make the folder searchable for python in your `.bash_profile` or `.bashrc` depending on your system

`export PYTHONPATH="$PYTHONPATH:/full/path/of/folder/containing/ExoTR:"`

Download the folders "PHO_STELLAR_MODEL" and "forward_rocky_mod" from the following link : [Google Drive](https://drive.google.com/drive/folders/1CQutXQ8Ki59TB9Dndo61sktwS3uOM7qZ?usp=sharing) .
Place the downloaded folders inside the ExoReL folder.

## Usage
You have to prepare the "retrieval_example.dat" and "forward_example.dat" parameters files before running ExoTR. Refer to the examples provided for guidance.
The full list of possible parameters are listed in the "standard_parameters.dat" file, placed inside the ExoTR package. Do not modify the "standard_parameters.dat" file.

You can generate a transmission spectrum by typing in a python instance or script.py file the following lines:

`import ExoReL`

`spec = ExoReL.CREATE_SPECTRUM()`

`spec.run_forward('forward_example.dat')`

You can run a retrieval by typing in a python instance or script.py file the following lines:

`import ExoReL`

`ret = ExoReL.RETRIEVAL()`

`ret.run_retrieval('retrieval_example.dat')`

To run the retrieval mode you need to have the MultiNest libraries installed in your system as well as `pymultinest (v2.11)`.

`pymultinest` is MPI compatible, therefore, you can run ExoTR to perform the sampling of the retrieval in parallel (you will need to install `mpi4py`):

`mpirun -np 10 python exotr_retrieval.py`

## Plotting the results
The plotting of the retrieval results is automatic and will produce the following graphs:
* Chemistry of the atmosphere versus the atmospheric pressure;
* Mean molecular mass versus the atmospheric pressure;
* The input spectral data and the best fit model calculated by the Bayesian sampling;
* The spectral contribution plot;
* The traces of the fitted free parameters;
* The posterior distribution corner plot.

In case `pymultinest` finds multiple solutions, ExoTR will automatically plot the aforementioned graphs for each of the solutions.

## Code usage in literature

* Hu 2019, [Information in the Reflected-light Spectra of Widely Separated Giant Exoplanets](https://iopscience.iop.org/article/10.3847/1538-4357/ab58c7), ApJ, 887, 166.
* Damiano & Hu 2020, [ExoReL<sup>R</sup>: A Bayesian Inverse Retrieval Framework for Exoplanetary Reflected Light Spectra](https://iopscience.iop.org/article/10.3847/1538-3881/ab79a5), AJ, 159, 175.
* Damiano et al. 2020, [Multi-orbital-phase and Multiband Characterization of Exoplanetary Atmospheres with Reflected Light Spectra](https://iopscience.iop.org/article/10.3847/1538-3881/abb76a), AJ, 160, 206.
* Damiano & Hu 2021, [Reflected Spectroscopy of Small Exoplanets I: Determining the Atmospheric Composition of Sub-Neptunes Planets](https://iopscience.iop.org/article/10.3847/1538-3881/ac224d), AJ, 162, 200.
* Damiano & Hu 2022, [Reflected Spectroscopy of Small Exoplanets II: Characterization of Terrestrial Exoplanets](https://iopscience.iop.org/article/10.3847/1538-3881/ac6b97), AJ, 163, 299.
* Damiano et al. 2023, [Reflected Spectroscopy of Small Exoplanets III: probing the UV band to measure biosignature gasses](https://iopscience.iop.org/article/10.3847/1538-3881/acefd3), AJ, 166, 157.
* Tokadjian et al. 2024, [The Detectability of CH<sub>4</sub>/CO<sub>2</sub>/CO and N<sub>2</sub>O Biosignatures Through Reflection Spectroscopy of Terrestrial Exoplanets](https://iopscience.iop.org/article/10.3847/1538-3881/ad88eb), AJ, 168, 292.
* Damiano et al. 2025, [Effects of planetary mass uncertainties on the interpretation of the reflectance spectra of Earth-like exoplanets](https://iopscience.iop.org/article/10.3847/1538-3881/ada610), AJ, 169, 97.

## Acknowledgement
The research was carried out at the Jet Propulsion Laboratory, California Institute of Technology, under a contract with the National Aeronautics and Space Administration (80NM0018D0004).
The High Performance Computing resources used in this investigation were provided by funding from the JPL Information and Technology Solutions Directorate.

## License
Copyright © 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Licensed under the Apache License, Version 2.0 (the "Licence");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## How to cite ExoReL
Please refer to the CITATION file. It contains the specific citations to reference the ExoReL framework in your publication.

## BUGS!!!
For any issues and bugs please send an e-mail at [mario.damiano@jpl.nasa.gov](mario.damiano@jpl.nasa.gov), or submit an issue through the GitHub system.
