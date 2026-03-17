# -*- coding: utf-8 -*-
import mne
from os import getcwd
from os import path, makedirs, environ
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def prepare_source_space(p):
    print('Preparing source space: started')
    cwd = getcwd()
   
    lh_surface_file = path.join(cwd, p.subj_brains_dir,
                             p.fs_subject_dir,p.surf_dir,'lh.white')
    rh_surface_file = path.join(cwd, p.subj_brains_dir,
                             p.fs_subject_dir,p.surf_dir,'rh.white')
    
    dists = [5,10,15,20]
    
    for dist in dists:
        src = mne.setup_source_space(
            subject= p.fs_subject_dir,
            spacing= dist, 
            surface="white",
            subjects_dir=p.subj_brains_dir,
            add_dist=False
        )

        srcspacepath = path.join(cwd, p.data, p.processed_data,p.protocol,
                  p.subj, p.session, p.eeg_dir, f'space_{dist}mm-src.fif')

        src.save(srcspacepath, overwrite=True)
    
    print('Preparing source space: completed')
    

#----------------------------------------------------------------------------------#
def prepare_leadfield(p):
    print('Preparing leadfield: started')
    cwd = getcwd()
    
    # Load raw
    rawpath = path.join(cwd, p.data, p.processed_data,p.protocol,
              p.subj, p.session, p.eeg_dir, 'subject-epo.fif')
    raw = mne.read_epochs(rawpath, preload=True)
    info = raw.info
    
    # Load BEM
    bemfullpath = path.join(cwd, p.subj_brains_dir,
                             p.fs_subject_dir,'bem','3shellbem.h5')
    bem = mne.read_bem_solution(bemfullpath)
    
    # Load trans file
    transsavepath = path.join(cwd, p.data, p.processed_data,p.protocol,
              p.subj, p.session, p.eeg_dir, 'template-trans.fif')
    
    montage = mne.channels.make_standard_montage("standard_1005")
    trans = mne.channels.compute_native_head_t(montage)
    mne.write_trans(transsavepath, trans, overwrite = True)
    
    # Better transformation with scaling - this should bring electrodes closer
    trans = mne.channels.compute_native_head_t(
        montage, 
        on_missing='ignore',  # Ignore missing channels
        verbose=True
    )
    
    mne.write_trans(transsavepath, trans, overwrite=True)
    

    dists = [5,10,15,20]
    
    for dist in dists:
        # Load source space    
        srcspacepath = path.join(cwd, p.data, p.processed_data,p.protocol,
                  p.subj, p.session, p.eeg_dir, f'space_{dist}mm-src.fif')
        src = mne.read_source_spaces(srcspacepath)

        # Make forward solution
        fwd = mne.make_forward_solution(info, trans=trans, 
                                        src=src, 
                                        bem=bem, 
                                        meg=False, 
                                        eeg=True, 
                                        mindist=5.0,
                                        n_jobs=1)

        # Save forward solution
        fwdsavepath = path.join(cwd, p.data, p.processed_data,p.protocol,
                  p.subj, p.session, p.leadfield_dir, f'subject_{dist}-fwd.fif')

        mne.write_forward_solution(fwdsavepath, fwd, overwrite=True)
    
        mne.viz.plot_alignment(info, trans=trans, 
                               subject= p.fs_subject_dir, 
                               subjects_dir=p.subj_brains_dir, 
                               src=src, 
                               bem=bem, show_axes=True, dig=True)

    
    print('Preparing leadfield: completed')