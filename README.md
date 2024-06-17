This repository holds scripts to perform fragment correlation mass spectrometry (FCMS).

The method is described in the following work:

Li Y, Cavet GL, Zare RN, Driver T. Fragment correlation mass spectrometry: determining the structures of biopolymers in a complex mixture without isolating individual components. ChemRxiv. 2024; doi:10.26434/chemrxiv-2024-pl3t1

The scripts are written for Python 2.7. To perform the analysis, run 'run_FCMS.py'.
The input is an mgf file of the MS/MS scan under analysis.
There should be no peak picking performed in the conversion to mgf.
The output is a list of correlation features with corresponding peak volumes and correlation score.
The output is saved as a .npy file.

This work is licensed under CC BY-NC-SA 4.0
To enquire about licensing opportunities, please contact the author: tarandriver(at)gmail.com
