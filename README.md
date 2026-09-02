![alt text](https://github.com/MDamiano/ExoReL/blob/main/ExoReL_logo.png?raw=true)

# ExoReL<sup>R</sup>

<a href="https://emac.gsfc.nasa.gov?cid=2601-028"><img src="https://img.shields.io/badge/EMAC-2601--028-blue" alt="EMAC 2601-028"></a>
<a href="https://ascl.net/2602.013"><img src="https://img.shields.io/badge/ascl-2602.013-blue.svg?colorB=262255" alt="ascl:2602.013" /></a>

Includes:
* A routine to generate single reflection and emission spectrum with and without noise and errorbars;
* A routine to generate datasets of reflection/emission spectra using Sobol or random sequencing;
* A retrieval routine based on nested sampling (i.e. MultiNest).

## Authors
* [Mario Damiano](https://mdamiano.github.io/) (Jet Propulsion Laboratory, California Institute of Technology)
* [Renyu Hu](https://renyuplanet.github.io/) (Rutgers, The State University of New Jersey)

## Collaborators
* Armen Tokadjian (Jet Propulsion Laboratory, California Institute of Technology)
* Zachary Burr (ETH Zurich)

## Installation:
Navigate to the desired location for the ExoReL folder. Therefore clone the code via `git`:

`git clone https://github.com/MDamiano/ExoReL.git`

Navigate into the downloaded ExoReL folder. Activate yout preferred python environment. Install python packages dependency:

`pip install -r requirements.txt`

Do not forget to make the folder reachable by python in your `.bash_profile` or `.bashrc` depending on your system

`export PYTHONPATH="$PYTHONPATH:/full/path/of/folder/containing/ExoReL:"`

After installation, upon the first import of the package, ExoReL will automatically download required files. The forward-model download generally takes less than a minute.

It will download the "forward_mod" folder from the following link: [Google Drive](https://drive.google.com/drive/folders/1CQutXQ8Ki59TB9Dndo61sktwS3uOM7qZ?usp=sharing). The required folder will be placed inside the ExoReL folder.

Stellar flux calculations use the PHOENIX grid distributed for `stsynphot`. On
first import, ExoReL downloads this approximately 1.8 GB atlas from the
[STScI reference-atlas archive](https://archive.stsci.edu/hlsp/reference-atlases)
into `ExoReL/stsynphot_data/grid/phoenix` and configures `PYSYN_CDBS`
automatically. The local data directory is ignored by Git. You can also trigger
or retry the one-time setup explicitly with `ExoReL.setup_stsynphot_data()`.

If `PYSYN_CDBS` already points to a complete external atlas, ExoReL reuses it
instead of downloading a duplicate. A complete atlas must contain
`grid/phoenix/catalog.fits` and the PHOENIX model files.

For molecular-opacity grids above resolving power `R=10000`, higher-resolution
BT-Settl spectra can be downloaded manually from the
[SVO Theory Server](https://svo2.cab.inta-csic.es/theory/newov2/index.php). Place
the downloaded ASCII spectra together in a directory and set `stellar_dir` in
the ExoReL JSON parameter file. ExoReL reads the `teff`, `logg`, and `meta`
values from each file header and performs bounded trilinear flux interpolation
in `(teff, logg, meta)` through `synphot`. The directory must contain every
bounding corner model required for the requested stellar parameters; ExoReL
does not silently extrapolate or substitute the nearest model. The source files
and interpolation weights are recorded in `starfx_model`. ExoReL validates the
SVO column units (Angstrom and erg cm⁻² s⁻¹ Angstrom⁻¹), converts SVO optical air
wavelengths to the vacuum frame used by the forward model, and conserves flux
density during that coordinate transformation. If `use_float32` is enabled,
the returned stellar wavelength and flux arrays are stored as `float32`; native
high-resolution files are still parsed at `float64` precision before rebinning.

The Python radiative-transfer branch also evaluates this same selected stellar
model directly on the molecular-opacity grid. Its surface flux density is
converted from W m⁻² µm⁻¹ to W m⁻² nm⁻¹ and diluted to the planet by
`(Rs * R_sun / (major-a * au))²`. The observational `starfx` arrays remain
separate, bin-integrated quantities. The C radiative-transfer branch continues
to use its legacy `solar0.txt` illumination spectrum.

```json
"stellar_dir": "/path/to/downloaded/SVO/stellar_spectra"
```

## Usage
You have to prepare the "retrieval_example.dat" and "forward_example.dat" parameters files before running ExoReL.

You can generate a transmission spectrum by typing in a python instance or script.py file the following lines:

`import ExoReL`

`spec = ExoReL.CREATE_SPECTRUM()`

`spec.run_forward('forward_example.dat')`

You can run a retrieval by typing in a python instance or script.py file the following lines:

`import ExoReL`

`ret = ExoReL.RETRIEVAL()`

`ret.run_retrieval('retrieval_example.json')`

To run the retrieval mode you need to have the MultiNest libraries installed in your system as well as `pymultinest (v2.11)`.

`pymultinest` is MPI compatible, therefore, you can run ExoReL to perform the sampling of the retrieval in parallel (you will need to install `mpi4py`):

`mpirun -np 10 python exorel_retrieval.py`

## Plotting the results
The plotting of the retrieval results can be automatic when chosen in the input parameter file and will produce the following graphs:
* Chemistry of the atmosphere versus the atmospheric pressure;
* Mean molecular mass versus the atmospheric pressure;
* The input spectral data and the best fit model calculated by the Bayesian sampling;
* The spectral contribution plot;
* The traces of the fitted free parameters;
* The posterior distribution corner plot;
* Expected log-pointwise probability (elpd) statistics abd leave-one-out (loo) analysis.

In case `pymultinest` finds multiple solutions, ExoReL will automatically plot the aforementioned graphs for each of the solutions.

## Code usage in literature

* Hu 2019, [Information in the Reflected-light Spectra of Widely Separated Giant Exoplanets](https://iopscience.iop.org/article/10.3847/1538-4357/ab58c7), ApJ, 887, 166.
* Damiano & Hu 2020, [ExoReL<sup>R</sup>: A Bayesian Inverse Retrieval Framework for Exoplanetary Reflected Light Spectra](https://iopscience.iop.org/article/10.3847/1538-3881/ab79a5), AJ, 159, 175.
* Damiano et al. 2020, [Multi-orbital-phase and Multiband Characterization of Exoplanetary Atmospheres with Reflected Light Spectra](https://iopscience.iop.org/article/10.3847/1538-3881/abb76a), AJ, 160, 206.
* Damiano & Hu 2021, [Reflected Spectroscopy of Small Exoplanets I: Determining the Atmospheric Composition of Sub-Neptunes Planets](https://iopscience.iop.org/article/10.3847/1538-3881/ac224d), AJ, 162, 200.
* Damiano & Hu 2022, [Reflected Spectroscopy of Small Exoplanets II: Characterization of Terrestrial Exoplanets](https://iopscience.iop.org/article/10.3847/1538-3881/ac6b97), AJ, 163, 299.
* Damiano et al. 2023, [Reflected Spectroscopy of Small Exoplanets III: probing the UV band to measure biosignature gasses](https://iopscience.iop.org/article/10.3847/1538-3881/acefd3), AJ, 166, 157.
* Tokadjian et al. 2024, [The Detectability of CH<sub>4</sub>/CO<sub>2</sub>/CO and N<sub>2</sub>O Biosignatures Through Reflection Spectroscopy of Terrestrial Exoplanets](https://iopscience.iop.org/article/10.3847/1538-3881/ad88eb), AJ, 168, 292.
* Damiano et al. 2025, [Effects of planetary mass uncertainties on the interpretation of the reflectance spectra of Earth-like exoplanets](https://iopscience.iop.org/article/10.3847/1538-3881/ada610), AJ, 169, 97.
* Burr et al. 2026, [Retrieving the Red Edge on Earth-like Planets with Heterogeneous Clouds and Surfaces](https://iopscience.iop.org/article/10.3847/1538-4357/ae563d), ApJ, 1001, 222.

## Acknowledgement
The research was carried out at the Jet Propulsion Laboratory, California Institute of Technology, under a contract with the National Aeronautics and Space Administration (80NM0018D0004).
The High Performance Computing resources used in this investigation were provided by funding from the JPL Information and Technology Solutions Directorate.

## License
Copyright © 2026, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

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
