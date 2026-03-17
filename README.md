# SNR_Optimization
This pipeline measures and compares the amplitude based SNR in sensor and source spaces for sensory evoked potentials.

Requirements and expected folder structure are stated in the SNR_Optimization.ipynb. 

The dataset that accompanies this script can be found in Zenodo, with DOI: 10.5281/zenodo.19063904. The zipped file containes the processed_data folder and the subject_1 folder. The subject_1 folder contains the head model that was used in this study and needs to be placed in the root directory like that. The processed_data folder contains the eeg, leadfield and current folders for all participants. It also needs to be placed in the root directory. 

The pre-processed EEG epochs for all participants can be found in the eeg folder of each participant. 

The second part of the script produces the necessary figures. If all eeg and current data exist in the folders, there is no need to run the first section of the script. 




