# -*- coding: utf-8 -*-
import mne
from os import getcwd
from os import path
import nibabel as nib
from nilearn.datasets import fetch_icbm152_2009
from nilearn import plotting
import numpy as np


def manually_fix_bem(p):
    cwd = getcwd()
    
    vertices, faces = mne.read_surface(path.join(cwd,
                                                     p.subj_brains_dir,
                                                     p.fs_subject_dir,
                                                     'bem',
                                                     'inner_skull.surf'))
             
    z_threshold = np.percentile(vertices[:, 2], 5) #percentile
             
    center_x, center_y = np.mean(vertices[:, 0]), np.mean(vertices[:, 1])
    radius = 50  # Adjust based on your dataset
                
    mask = (vertices[:, 2] < z_threshold) & \
                ((vertices[:, 0] - center_x)**2 + (vertices[:, 1] - center_y)**2 < radius**2)
    
    vertices[mask, 2] += 15  # Move selected region up by 15 mm
          
    mne.write_surface(path.join(cwd,
                                p.subj_brains_dir,
                                p.fs_subject_dir,
                                'bem',
                                'inner_skull_fixed.surf'), vertices,faces, overwrite=True)

    

def load_head(p):
    print('Loading head template: started')
    cwd = getcwd()
    fullpath_brains = path.join(cwd, p.subj_brains_dir)
    
    conductivity = [0.33, 0.0042, 0.33]  # Standard for scalp, skull, brain (from fieldtrip)

    mne.viz.plot_bem(subject=p.fs_subject_dir, subjects_dir=fullpath_brains, orientation='coronal')
    bem_model = mne.make_bem_model(subject=p.fs_subject_dir, ico=4, conductivity=conductivity, subjects_dir=fullpath_brains)
    bem = mne.make_bem_solution(bem_model)
    
    savefullpath = path.join(cwd, p.subj_brains_dir,
                             p.fs_subject_dir,'bem','3shellbem.h5')
    
    mne.write_bem_solution(savefullpath, bem, overwrite=True)
    
    print('Loading head template: completed')