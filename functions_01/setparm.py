# -*- coding: utf-8 -*-

class Pars:
    def __init__(self, subj, s, protocol):
        
        # General parameters
        self.subj = subj
        self.session = s
        self.protocol = protocol
        
        # Folder names
        self.eeg_dir = 'eeg'
        self.leadfield_dir = 'leadfield'
        self.trial_dir = 'trial'
        self.roi_current_dir = 'roi_current'
        self.data = 'data'
        self.raw_data = 'raw_data'
        self.processed_data = 'processed_data'
        self.subj_brains_dir = 'subj_brains'
        self.fs_subject_dir = 'subject_1'
        self.bem_model_dir = 'bem'
        self.surf_dir = 'surf'
        self.current_dir = 'current'
        self.roi_current_dir = 'roi_current'
        self.average_connectome_dir = 'avg_connectome_MATLAB'
        self.delay_ms_file = 'delay_ms_matrix'
        self.dynamics_dir = 'dynamics'
        
        # File names
        self.epochsfile = 'subject-epo.fif'
        
        # EEG parameters
        self.sep_window_lo = 1
        self.sep_window_hi = 1
        self.task_list = 1
        self.sampling_rate = 2048
        self.electrode_list = list(range(128))
        
        # For keeping data around triggers
        self.start_lat = -1
        self.stop_lat = 2
        
        # Artifact removal
        self.artifact_removal = False
        self.artifact_start = -0.005
        self.artifact_end = 0.005
        
        # Filtering parameters
        self.highpass_filter  = 1 #(Hz)
        self.lowpass_filter   = 80
        self.notch_filter = 50;   
        
        # mne.preprocessing.ICA
        self.downsamplefreq = 512
        
        # Create mne.Epochs 
        self.event_dict = {
                            '1' : 1
                            }
        self.tstart = -0.1 #(s)
        self.tstop  = 0.2
        
        # Epoch ID for erp
        self.selected_epoch = '1'
        
        
        # Source Localization parameters
        self.reps = [125,250,500,1000,1500,2000]  
        self.reps_split =  [65,125,250,500,750,1000] 
        self.dists = [5,10,15,20]
        self.methods = ['eLORETA', 'sLORETA', 'dSPM', 'MNE']
        
        # SNR parameters
        self.tmin_snr = 0.02
        self.tmax_snr = 0.06
        self.tmin_base_snr = -0.1
        self.tmax_base_snr = 0.0
        self.subjects = ['01','02','03', '04', '05', '08','09', '10', '11', '12']

        # Estimate dynamics 
        self.lamda = 1e-4
        self.order_auto = 2
        self.conduction_velocity = 6.0
        self.self_delay = 27
        
        
        
              
def set_parm( subject, session, protocol):
    print('Setting up parameters: started')
    p = Pars(subject,session,protocol)
    print('Setting up parameters: completed')
    return p