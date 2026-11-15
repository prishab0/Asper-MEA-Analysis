# References

Data and methods used in this repository. Every dataset below is public. Cite the original authors, not this repository.

## Data

1. Wagenaar DA, Pine J, Potter SM. An extremely rich repertoire of bursting patterns during the development of cortical cultures. *BMC Neuroscience* 2006;7:11. doi:10.1186/1471-2202-7-11. Data: potterlab.bme.gatech.edu/development-data/
2. Vinogradov et al. Bicuculline (blocked inhibition) recordings of mouse cortical cultures, two plates (day 17 and day 18). bioRxiv 2024. doi:10.1101/2024.08.21.608974. Dataset downloaded by the burst-shape tutorial from figshare. *Author list and title to be completed from the preprint.*
3. Lamberti M, van Putten M, Marzen S, le Feber J. NMDA receptors involvement in memory and prediction. Dryad, 2024. doi:10.5061/dryad.qz612jmpv. The scripts use the folder `Test APV concentration`. Associated paper: Lamberti M, van Putten MJAM, Marzen S, le Feber J. The role of NMDA receptors in memory and prediction in cultured neural networks. *Journal of Neural Engineering* 2025, doi:10.1088/1741-2552/adb5c5 (preprint: bioRxiv doi:10.1101/2024.02.01.578348).

## Methods and software

4. Schäfer TJ, Giannakakis E, Schmidt-Barbo P, Levina A, Vinogradov O. Data-driven burst shape analysis for functional phenotyping of neuronal cultures. bioRxiv 2025. doi:10.1101/2025.09.29.679256. Code: github.com/LevinaLab/burst-shape (MIT licence), archived at doi:10.5281/zenodo.21035071. The scripts here run that pipeline unmodified and add the checks described in the report. The authors have not reviewed this work.
5. van Hugte EJH, Lewerissa EI, Wu KM, et al. SCN1A-deficient excitatory neuronal networks display mutation-specific phenotypes. *Brain* 2023;146(12):5153-5167. doi:10.1093/brain/awad245. *Check against the burst-shape repository's reference for the "traditional" summary statistics before relying on this attribution.*
6. Ciba M, Petzold M, Alves CL, Rodrigues FA, Jimbo Y, Thielemann C. Machine learning and complex network analysis of drug effects on neuronal microelectrode biosensor data. *Scientific Reports* 2025;15:15128. doi:10.1038/s41598-025-99479-7. Code: github.com/ManuelCiba/spike-train-ml-bic.

## Related work on decoding stimulus information from cultured networks

7. Dranias MR, Ju H, Rajaram E, VanDongen AMJ. Short-term memory in networks of dissociated cortical neurons. *Journal of Neuroscience* 2013;33(5):1940-1953. doi:10.1523/JNEUROSCI.2718-12.2013.
8. Ju H, Dranias MR, Banumurthy G, VanDongen AMJ. Spatiotemporal memory is an intrinsic property of networks of dissociated cortical neurons. *Journal of Neuroscience* 2015;35(9):4040-4051. doi:10.1523/JNEUROSCI.3793-14.2015.

This is not a systematic literature review. Corrections and additions are welcome.

## Software

numpy, pandas, SciPy, scikit-learn and matplotlib. A random seed of 0 is used throughout unless a script states otherwise.
