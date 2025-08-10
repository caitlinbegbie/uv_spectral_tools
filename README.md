# Tools to Analyze STIS Sun-like UV Spectral Products

## Summary
Functions and code written during STScI's 2025 Space Astronomy Summer Program designed to normalize and cross-correlate raw spectral products 
from the Mikulski Archive for Space Telescopes (MAST) into a comparable format. Focus is given to Sun-like stars in the ultraviolet range and
measured with the E230H grating.

## Folders and Files
* `data` folder: houses all of the MAST data used for cross-correlation shifting (`shifting_data` sub-directory) and >40 Sun-like stars for analysis (`sunlike_data`)
* `cross_correlate.py`: Gives functions used to cross-correlate a sample star against a basis star (C/O Sten Hasselquist).
* `shifting_example_notebook.ipynb`: Notebook demonstrating the use of the overall `spectra_to_rest` function, that takes in a stellar .fits file and
  returns air rest shifted wavelengths and normalized flux values.
* `spectra_to_rest.py`: Gives `spectra_to_rest` function, and the background functions used to reduce data into an analyzable format.
* `spectral_analysis_example_notebook.ipynb`: Notebook exploring relationships between stellar parameters and spectral features in Sun-like stars.
