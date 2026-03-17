# -*- coding: utf-8 -*-
import mne
import os
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from scipy.linalg import svd


#------------------------------------------------------------------------------------------------------#
def eeg_preica(p):
    print('Preprocessing: started')
    
    cwd = os.getcwd()
    rawdatapath = os.path.join(cwd, p.data, p.raw_data, p.protocol, p.subj, p.session)
    
    for file in os.listdir(rawdatapath):
        if file.endswith('.cnt'):
            fullpath_raw = os.path.join(rawdatapath,file) 
            raw = mne.io.read_raw_ant(fullpath_raw,preload=False)
        elif file.endswith('.set'):
            fullpath_raw = os.path.join(rawdatapath,file)
            raw = mne.io.read_raw_eeglab(fullpath_raw,preload=False)
        elif file.endswith('.elc'):
            fullpath_elc = os.path.join(rawdatapath,file) 
        
    
    # Load raw data 
    raw_cropped = raw.crop()
    raw_cropped.load_data()
   
    # Only keep channels of interest
    print("Before dropping:", len(raw_cropped.ch_names))  

    zero_channels = [ch for ch, data in zip(raw_cropped.ch_names, raw_cropped.get_data()) if np.all(data == 0)]
    print("Detected zeroed EEG channels:", zero_channels)
    raw_cropped.drop_channels(zero_channels)

    data, _ = raw_cropped[:]
    std_per_channel = np.std(data, axis=1)
    threshold = 5 * np.median(std_per_channel)
    bad_channels = np.array(raw_cropped.ch_names)[std_per_channel > threshold]
    print("Detected bad EEG channels:", bad_channels)
    raw_cropped.info["bads"].extend(bad_channels.tolist())
    raw_cropped.drop_channels(raw_cropped.info["bads"]) 
    
    print("After dropping:", len(raw_cropped.ch_names)) 
    
    # Load sensor locations
    '''
    mntg = mne.channels.read_custom_montage(fullpath_elc)
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    mtrx = np.eye(4)  
    mtrx[:3, :3] = rotation  
    mtrx[:3, :3] *= 1.3
    trans = mne.transforms.Transform("unknown", "head", mtrx) 
    mntg.apply_trans(trans)
    '''
    raw_cropped.set_montage('standard_1005')
    
    # Bandpass filter
    raw_cropped.filter(l_freq=p.highpass_filter, h_freq=p.lowpass_filter)
    # Remove noise from power line
    raw_cropped.notch_filter(freqs=p.notch_filter)
    
    # Print dataset info
    print('----- Dataset Info -----')
    print(raw_cropped.info)
    
    #Save before applying ICA
    file = 'pre_ica.set'
    fullpath_preica = os.path.join(rawdatapath,file)
    mne.export.export_raw(fullpath_preica,raw_cropped,overwrite=True)     
    print('Preprocessing: completed')
    
    
#------------------------------------------------------------------------------------------------------#    
def eeg_ica(p):
       
    cwd = os.getcwd()
    rawdatapath = os.path.join(cwd, p.data, p.raw_data, p.protocol, p.subj, p.session)
  
    file = 'pre_ica.set'
    fullpath_preica = os.path.join(rawdatapath,file)
    raw = mne.io.read_raw_eeglab(fullpath_preica,preload=True)
    
    
    # Downsample before ICA
    raw.resample(sfreq=p.downsamplefreq) #2048 -> 512
    # Highpass filter
    raw.filter(l_freq=1.0, h_freq=None)
    
    data = raw.get_data(picks="eeg")
    U, S, Vt = svd(data, full_matrices=False)  # S contains singular values

    # Plot singular values (log scale)
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(S) + 1), S, "o-", label="Singular values")
    plt.yscale("log")  # Log scale to show differences better
    plt.xlabel("Component Number")
    plt.ylabel("Singular Value (log scale)")
    plt.title("Singular Values of EEG Data (Before ICA)")
    plt.legend()
    plt.grid()
    plt.show()
    
    
    # Run ICA
    n_components = len(raw.ch_names)-3 #-1
    method = "fastica"  # "fastica", "infomax", "picard" needs scikit-learn for fastica
    random_state = 11   # For reproducibility

    ica = mne.preprocessing.ICA(n_components=n_components, 
                                method=method, 
                                random_state=random_state,
                                max_iter=1000)
    ica.fit(raw)
    
    #Save after applying ICA
    file = 'ica.set'
    fullpath_ica = os.path.join(rawdatapath,file)
    ica.save(fullpath_ica, overwrite=True) 
    
 
#------------------------------------------------------------------------------------------------------#
def eeg_postica(p):
    cwd = os.getcwd()
    rawdatapath = os.path.join(cwd, p.data, p.raw_data, p.protocol, p.subj, p.session)
    processeddatapath = os.path.join(cwd, p.data, p.processed_data, p.protocol, p.subj, p.session, p.eeg_dir)

    for file in os.listdir(rawdatapath):
        if file.endswith('.trg'):
            fullpath_trg = os.path.join(rawdatapath,file)
    
    file = 'pre_ica.set'
    fullpath_preica = os.path.join(rawdatapath,file)
    
    file = 'ica.set'
    fullpath_ica = os.path.join(rawdatapath,file)
    
    raw = mne.io.read_raw_eeglab(fullpath_preica,preload=True)
    ica = mne.preprocessing.read_ica(fullpath_ica)
    
    # Show some components in time
    raw_subset = raw.copy().crop(tmin=0, tmax=40)    
    raw_subset.set_annotations(None)
    sources = ica.get_sources(raw_subset).get_data()
    powers = np.var(sources, axis=1)
    sorted_inds = np.argsort(powers)[::-1]
    # Plot components in ranked order
    ica.plot_components(picks=sorted_inds[40:79], inst=raw_subset)
    
    ica.plot_sources(raw_subset,picks=sorted_inds[40:79]) 
    del raw_subset  # Remove unnecessary copy
    
    eog_indices, eog_scores = ica.find_bads_eog(raw,ch_name='Fp2') #Fp1
    ica.exclude = eog_indices
    ica.apply(raw)
    
    file = 'postica.set'
    fullpath_postica = os.path.join(rawdatapath,file)
    mne.export.export_raw(fullpath_postica,raw,overwrite=True)
    
    events, event_id = mne.events_from_annotations(raw)
    
    print(events)
    #events = np.unique(events, axis=0) # remove duplicates
    
    mne.viz.plot_events(events)
   
    # Split in epochs
    epochs = mne.Epochs(raw, 
                        events, 
                        p.event_dict, 
                        tmin= p.tstart, 
                        tmax= p.tstop, 
                        preload=True
                        )
    
    # Save epochs in file
    file = 'subject-epo.fif'
    fullpath_epochs = os.path.join(processeddatapath,file)
    epochs.save(fullpath_epochs, overwrite=True)
    

def eeg_preprocess(p, pre, ica, post):
    
    if pre:
        eeg_preica(p)
    if ica:
        print('ICA: started')
        eeg_ica(p)
        print('ICA: completed')  
    if post: 
        eeg_postica(p)
    
    
    