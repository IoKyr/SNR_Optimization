# -*- coding: utf-8 -*-
import mne
from os import getcwd
from os import path
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from scipy.spatial import KDTree
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

#------------------------------------------------------------------------------------#
def source_localization(p):
    print('Source localization: started')
    
    cwd = getcwd()
    processedeegpath = path.join(cwd, p.data, p.processed_data, 
                                  p.protocol, p.subj, p.session, 
                                  p.eeg_dir)
    
    fullpath_epochs = path.join(processedeegpath,p.epochsfile)
    epochs = mne.read_epochs(fullpath_epochs)
    epochs.set_eeg_reference(projection=True)
    
    reps = p.reps
    dists = p.dists
    methods = p.methods
    
    for method in methods: 
        for rep in reps:     
            for dist in dists:
                # Load fwd solution
                fwdloadpath = path.join(cwd, p.data, p.processed_data,p.protocol,
                          p.subj, p.session, p.leadfield_dir, f'subject_{dist}-fwd.fif')

                fwd = mne.read_forward_solution(fwdloadpath)

                # Select subset of epochs and make erp
                selected_epochs = epochs[p.selected_epoch]   
                n_epochs = len(selected_epochs)
                n_select = rep
                np.random.seed(11)
                random_indices = np.random.choice(n_epochs, n_select, replace=False)
                random_subset = selected_epochs[random_indices]  
                erp = random_subset.average()
                info = epochs.info

                cov = mne.compute_covariance(random_subset, tmax=0)
                #cov.plot(epochs.info, proj=True)

                inverse_operator = mne.minimum_norm.make_inverse_operator(info, fwd, cov, fixed=True, loose=0, depth=0.8 ) #0.2,
                stc = mne.minimum_norm.apply_inverse(erp, inverse_operator, lambda2=1.0/9.0, method=method) 
                #method: “MNE” | “dSPM” | “sLORETA” | “eLORETA”

                stcsavepath = path.join(cwd, p.data, p.processed_data, 
                                              p.protocol, p.subj, p.session, 
                                              p.current_dir, f'current{rep}reps_{dist}mm_{method}')

                stc.save(stcsavepath, ftype='stc', overwrite=True, verbose=None)

    print('Source localization: completed')
    
#------------------------------------------------------------------------------------#    

def plot_stc(p):
    
    print('Generating stc plot: started')
    cwd = getcwd()
    
    reps = 2000
    dist = 5
    method = 'dSPM'
    
    stcloadpath = path.join(cwd, p.data, p.processed_data, 
                                  p.protocol, p.subj, p.session, 
                                  p.current_dir, f'current{reps}reps_{dist}mm_{method}-lh.stc')  
    stc = mne.read_source_estimate(stcloadpath)
    
    brain = stc.plot(subject=p.fs_subject_dir, surface='inflated', hemi='both', views='lat', 
             subjects_dir=p.subj_brains_dir, colormap='viridis')

    subjdirpath = path.join(cwd, p.subj_brains_dir)
    labels = mne.read_labels_from_annot(
        subject=p.fs_subject_dir,
        parc='HCPMMP1',
        hemi='both',
        subjects_dir=subjdirpath
        )
    # Add labels to the brain plot
    s1_names = ['L_3b_ROI-lh', 'L_1_ROI-lh', 'L_2_ROI-lh', 'R_3b_ROI-rh', 'R_1_ROI-rh', 'R_2_ROI-rh']
    s1_labels = [label for label in labels if label.name in s1_names]
    for label in s1_labels:
        brain.add_label(label, borders=True, color='red', alpha=0.6)
    print('Generating stc plot: completed')
    
    
#------------------------------------------------------------------------------------#
def source_localization_split(p):
    print('Source localization with split: started')
    
    cwd = getcwd()
    processedeegpath = path.join(cwd, p.data, p.processed_data, 
                                  p.protocol, p.subj, p.session, 
                                  p.eeg_dir)
    
    fullpath_epochs = path.join(processedeegpath, p.epochsfile)
    epochs = mne.read_epochs(fullpath_epochs)
    epochs.set_eeg_reference(projection=True)
    
    reps = p.reps_split
    dists = p.dists
    methods = p.methods
    
    selected_epochs = epochs[p.selected_epoch]
    n_epochs = len(selected_epochs)
    
    # Split epochs into two halves
    midpoint = n_epochs // 2
    first_half = selected_epochs[:midpoint]
    second_half = selected_epochs[midpoint:]
    
    for method in methods: 
        for rep in reps:     
            for dist in dists:
                # Load fwd solution
                fwdloadpath = path.join(cwd, p.data, p.processed_data, p.protocol,
                                        p.subj, p.session, p.leadfield_dir, f'subject_{dist}-fwd.fif')

                fwd = mne.read_forward_solution(fwdloadpath)

                for i, half in enumerate([first_half, second_half], start=1):
                    n_select = rep
                    np.random.seed(11 + i)  # different seed for each half
                    random_indices = np.random.choice(len(half), n_select, replace=False)
                    random_subset = half[random_indices]
                    erp = random_subset.average()
                    info = epochs.info
                    
                    cov = mne.compute_covariance(random_subset, tmax=0)

                    inverse_operator = mne.minimum_norm.make_inverse_operator(info, fwd, cov, fixed=True, loose=0, depth=0.8 ) #0.2,
                    stc = mne.minimum_norm.apply_inverse(erp, inverse_operator, lambda2=1.0/9.0, method=method)

                    stcsavepath = path.join(cwd, p.data, p.processed_data, 
                                            p.protocol, p.subj, p.session, 
                                            p.current_dir, f'current{rep}reps_{dist}mm_{method}_split{i}')
                    
                    stc.save(stcsavepath, ftype='stc', overwrite=True, verbose=None)

    print('Source localization with split: completed')


#------------------------------------------------------------------------------------#
#                    PLOT FUNCTIONS    
#------------------------------------------------------------------------------------#        
def plot_same_source_different_methods(p, remove_outliers=False):
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    cwd = getcwd()
    reps = p.reps
    dists = [5, 10, 15, 20]
    methods = ['dSPM', 'sLORETA', 'eLORETA', 'MNE']
    
    # Get list of subjects - p.subj is single, p.subjects is the list
    print(f"Debug - hasattr(p, 'subjects'): {hasattr(p, 'subjects')}")
    if hasattr(p, 'subjects'):
        print(f"Debug - p.subjects: {p.subjects}")
        print(f"Debug - type(p.subjects): {type(p.subjects)}")
    print(f"Debug - p.subj: {p.subj}")
    
    if hasattr(p, 'subjects') and p.subjects is not None and len(p.subjects) > 0:
        subjects = p.subjects
        print(f"Using p.subjects: {subjects}")
    elif isinstance(p.subj, list):
        subjects = p.subj
        print(f"Using p.subj as list: {subjects}")
    else:
        subjects = [p.subj]
        print(f"Using p.subj as single subject: {subjects}")
    
    print(f"Loading data for {len(subjects)} subject(s): {subjects}")

    # ------------------ Load all STCs for all subjects ------------------
    stcs = {subj: {dist: {method: [] for method in methods} for dist in dists} 
            for subj in subjects}
    
    for subj in subjects:
        print(f"Loading STCs for subject: {subj}")
        files = [f"current{num}reps_{dist}mm_{method}-lh.stc"
                 for num, dist, method in product(reps, dists, methods)]
        stc_files = [
            path.join(cwd, p.data, p.processed_data, p.protocol, subj,
                      p.session, p.current_dir, filename)
            for filename in files
        ]

        for i, f in enumerate(stc_files):
            dist = dists[(i // len(methods)) % len(dists)]
            method = methods[i % len(methods)]
            stcs[subj][dist][method].append(mne.read_source_estimate(f))

    times = next(iter(stcs[subjects[0]][dists[0]].values()))[0].times

    # ------------------ SCALING: Scale all methods to pA/m² like MNE ------------------
    scaling_factors = {
        dist: {
            method: (
                max(np.max(np.abs(stc.data)) for stc in stcs[subjects[0]][dist]["MNE"])
                / max(np.max(np.abs(stc.data)) for stc in stcs[subjects[0]][dist][method])
            )
            for method in ["dSPM", "sLORETA", "eLORETA"]
        }
        for dist in dists
    }

    # ------------------ SNR CALCULATION ------------------
    def calculate_snr(stc, times, method, dist, scaling_factors):
        """Calculate SNR for each source: RMS(25-65ms) / std(baseline)"""
        # Scale the data to MNE pA/m²
        if method == "MNE":
            scaled_data = stc.data * 1e12
        else:
            scaled_data = stc.data * scaling_factors[dist][method] * 1e12
        
        # Time masks
        signal_mask = (times >= 0.025) & (times <= 0.065)
        baseline_mask = times < 0
        
        # Calculate RMS for signal window
        signal_rms = np.sqrt(np.mean(scaled_data[:, signal_mask] ** 2, axis=1))
        
        # Calculate std for baseline
        baseline_std = np.std(scaled_data[:, baseline_mask], axis=1)
        
        # SNR (avoid division by zero)
        snr = np.divide(signal_rms, baseline_std, 
                       out=np.zeros_like(signal_rms), 
                       where=baseline_std != 0)
        
        return snr

    # Find highest SNR source for each subject (using global scaling for SNR calculation)
    print("\nCalculating SNR for each subject...")
    highest_snr_sources = {}
    
    for subj in subjects:
        # Use 20mm distance and 2000 reps (highest quality) for SNR calculation
        reference_dist = 20
        reference_rep_idx = reps.index(2000)
        
        # Average SNR across all methods for this subject
        all_snrs = []
        for method in methods:
            stc = stcs[subj][reference_dist][method][reference_rep_idx]
            snr = calculate_snr(stc, stc.times, method, reference_dist, scaling_factors)
            all_snrs.append(snr)
        
        # Average SNR across methods
        avg_snr = np.mean(all_snrs, axis=0)
        max_snr_idx = np.argmax(avg_snr)
        
        highest_snr_sources[subj] = max_snr_idx
        print(f"Subject {subj}: highest SNR source = {max_snr_idx}, SNR = {avg_snr[max_snr_idx]:.2f}")

    # For Figure 1, use the first subject's highest SNR source
    fig1_subject = subjects[0]
    max_source_idx = highest_snr_sources[fig1_subject]
    print(f"\nFigure 1 will use subject {fig1_subject}, source {max_source_idx}")
    
    # Recalculate scaling factors based on the SNR-selected source (for fair comparison)
    # Use 20mm distance and 2000 reps as reference
    reference_dist = 20
    reference_rep_idx = reps.index(2000)
    
    source_based_scaling = {
        dist: {
            method: (
                np.max(np.abs(stcs[fig1_subject][dist]["MNE"][reference_rep_idx].data[max_source_idx]))
                / np.max(np.abs(stcs[fig1_subject][dist][method][reference_rep_idx].data[max_source_idx]))
            )
            for method in ["dSPM", "sLORETA", "eLORETA"]
        }
        for dist in dists
    }
    
    print(f"\nSource-specific scaling factors at {reference_dist}mm:")
    for method in ["dSPM", "sLORETA", "eLORETA"]:
        print(f"  {method}: {source_based_scaling[reference_dist][method]:.3f}")

    # Set consistent matplotlib parameters for both figures
    mpl.rcParams.update({
        'font.size': 30,
        'axes.titlesize': 30,
        'axes.labelsize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 24,
        'figure.dpi': 120
    })
    sns.set(style="whitegrid", font_scale=1.0)
    
    n_reps = len(reps)
    
    # ------------------ Figure 1: Scaled STC Activity (using highest SNR source) ------------------
    # Use more distinct grayscale values with different line styles
    
    # Repeat patterns if we have more reps than colors/styles
    gray_colors = ['#C0C0C0', '#C0C0C0', '#808080', '#808080', '#000000','#000000']
    line_styles = [':', '-', ':', '-', ':', '-']
    
    dist = 10
    subi = 0  # Use first subject
    df_list = []
    for method in methods:
        for i, stc in enumerate(stcs[subjects[subi]][dist][method]):
            if method == "MNE":
                scaled = stc.data[max_source_idx] * 1e12
            else:
                scaled = stc.data[max_source_idx] * source_based_scaling[dist][method] * 1e12
            stc_times = stc.times
            df_temp = pd.DataFrame({
                "Time (s)": stc_times,
                "Activity (pA/m²)": scaled,
                "Repetitions": reps[i],
                "Method": method
            })
            df_list.append(df_temp)
    df = pd.concat(df_list)


    font_scale = 1.5
    with sns.plotting_context("notebook", font_scale=font_scale):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    
        for idx, (ax, method) in enumerate(zip(axes, methods)):
            # Plot each repetition
            for i, rep in enumerate(reps):
                data = df[(df["Method"] == method) & (df["Repetitions"] == rep)]
                ax.plot(
                    data["Time (s)"], -data["Activity (pA/m²)"],
                    color=gray_colors[i],
                    linestyle=line_styles[i],
                    label=f"{rep} trials",
                    linewidth=2.5
                )
    
            # --- Titles & labels ---
            ax.set_title(f"{method}", weight='bold')
            ax.set_xlabel("Time (s)")
            if idx == 0:
                ax.set_ylabel("Current (pA/m²)", weight='bold')
            ax.set_xlim((-0.1, 0.2))
    
            # --- Spines & grid styling ---
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
    
            # Only show left spine for the first subplot
            if idx == 0:
                ax.spines['left'].set_visible(True)
            else:
                ax.spines['left'].set_visible(False)
    
            # Optional: make visible spines thinner and darker
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('black')
    
            # Remove y-axis ticks for non-left plots
            if idx != 0:
                ax.yaxis.set_visible(False)
    
            # Minimal grid
            ax.grid(False)
            ax.xaxis.grid(False, which='both')
            ax.yaxis.grid(False, which='both')
    
        # --- External legend at the top ---
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='upper center',
            ncol=len(reps),
            fontsize=12 * font_scale,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02)
        )
    
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # space for legend
        plt.show()


    # ------------------ Figure 2: Baseline Variance (using each subject's highest SNR source) ------------------
    baseline_var_data = []

    # Loop over all subjects, using their respective highest SNR source
    for subj in subjects:
        subj_source_idx = highest_snr_sources[subj]
        
        # Calculate subject-specific source-based scaling
        subj_source_scaling = {
            dist: {
                method: (
                    np.max(np.abs(stcs[subj][dist]["MNE"][reference_rep_idx].data[subj_source_idx]))
                    / np.max(np.abs(stcs[subj][dist][method][reference_rep_idx].data[subj_source_idx]))
                )
                for method in ["dSPM", "sLORETA", "eLORETA"]
            }
            for dist in dists
        }
        
        for dist in dists:
            for method in methods:
                for i, stc in enumerate(stcs[subj][dist][method]):
                    source_ts = stc.data[subj_source_idx]
                    stc_times = stc.times
                    if method == "MNE":
                        scaled_ts = source_ts * 1e12
                    else:
                        scaled_ts = source_ts * subj_source_scaling[dist][method] * 1e12
                    baseline_mask = stc_times < 0
                    baseline_var = np.var(scaled_ts[baseline_mask]) if np.any(baseline_mask) else np.nan
                    baseline_var_data.append({
                        'Subject': subj,
                        'Distance (mm)': dist,
                        'Method': method,
                        'Repetitions': reps[i],
                        'Baseline Power (pA²/m⁴)': baseline_var
                    })

    df_var = pd.DataFrame(baseline_var_data)
    method_order = methods
    df_var['Method'] = pd.Categorical(df_var['Method'], categories=method_order, ordered=True)
    
    print(f"\nTotal data points collected: {len(df_var)}")
    print(f"Unique distances: {sorted(df_var['Distance (mm)'].unique())}")
    print(f"Unique methods: {df_var['Method'].unique().tolist()}")
    print(f"Unique repetitions: {sorted(df_var['Repetitions'].unique())}")

    # --- Outlier handling ---
    print("\nChecking baseline variance distribution per method:")
    desc = df_var.groupby("Method")["Baseline Power (pA²/m⁴)"].describe()
    print(desc)

    # ------------------ NEW FIGURE: Baseline Variance distribution per method, trial, and distance ------------------
    print("\nPlotting Baseline Variance distribution per method, trial number, and distance (before outlier removal)...")

    if not df_var.empty:
        
        font_scale = 1.3
        
        with sns.plotting_context("notebook", font_scale=font_scale):
            n_methods = len(methods)
            n_dists = len(dists)
    
            fig, axes = plt.subplots(
                n_dists, n_methods,
                figsize=(5 * n_methods, 4 * n_dists),
                sharey=True, sharex=True
            )
    
            # Handle case where only one row or one column
            if n_dists == 1:
                axes = np.expand_dims(axes, axis=0)
            if n_methods == 1:
                axes = np.expand_dims(axes, axis=1)
    
            for i, dist in enumerate(dists):
                for j, method in enumerate(methods):
                    ax = axes[i, j]
                    data_sub = df_var[
                        (df_var["Method"] == method) &
                        (df_var["Distance (mm)"] == dist)
                    ]
    
                    sns.violinplot(
                        data=data_sub,
                        x="Repetitions",
                        y="Baseline Power (pA²/m⁴)",
                        inner=None,
                        color="lightgray",
                        ax=ax
                    )
                    sns.stripplot(
                        data=data_sub,
                        x="Repetitions",
                        y="Baseline Power (pA²/m⁴)",
                        hue="Subject",
                        dodge=True,
                        palette="tab10",
                        alpha=0.8,
                        size=5,
                        ax=ax
                    )
    
                    # --- Titles & labels ---
                    if i == 0:
                        ax.set_title(method, weight="bold", pad=10)
                    if j == 0:
                        ax.set_ylabel(f"Dist {dist} mm\nBaseline Var (pA²/m⁴)")
                    else:
                        ax.set_ylabel("")
    
                    if i == n_dists - 1:
                        ax.set_xlabel("Number of trials")
                    else:
                        ax.set_xlabel("")
    
                    # --- Aesthetic improvements ---
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(True)
    
                    # Only show left spine on first subplot in each row
                    if j == 0:
                        ax.spines['left'].set_visible(True)
                    else:
                        ax.spines['left'].set_visible(False)
                        ax.yaxis.set_visible(False)
    
                    # Optional: make visible spines thinner and darker
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.2)
                        spine.set_color('black')
    
                    # Minimal grid
                    ax.grid(False)
                    ax.xaxis.grid(False, which='both')
                    ax.yaxis.grid(False, which='both')
    
                    # --- Legend handling ---
                    if i == 0 and j == n_methods - 1:
                        ax.legend(
                            title="Subject",
                            fontsize=9,
                            title_fontsize=10,
                            loc='upper right'
                        )
                    else:
                        ax.get_legend().remove()
    
            plt.suptitle(
                "Baseline Variance per Method, Trial Number, and Distance (All Subjects)",
                weight='bold'
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    else:
        print("No baseline variance data available to plot.")


    def iqr_filter(df, col, factor=1.5):
        mask = pd.Series(True, index=df.index)
        for method in method_order:
            idx = df["Method"] == method
            vals = df.loc[idx, col]
            if vals.isna().all():
                continue
            q1, q3 = np.percentile(vals.dropna(), [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - factor * iqr, q3 + factor * iqr
            mask.loc[idx] = (vals >= lower) & (vals <= upper)
        return mask  

    if remove_outliers:
        mask = iqr_filter(df_var, "Baseline Power (pA²/m⁴)")
        before, after = len(df_var), mask.sum()
        print(f"Removed {before - after} outliers ({100*(1 - after/before):.1f}%) based on IQR rule.")
        df_var = df_var.loc[mask].reset_index(drop=True)
        
        print(f"\nAfter filtering - Total data points: {len(df_var)}")
        print(f"Distances: {sorted(df_var['Distance (mm)'].unique())}")
        print(f"Methods: {df_var['Method'].unique().tolist()}")
        print(f"Repetitions: {sorted(df_var['Repetitions'].unique())}")
    else:
        print("Keeping all data points (outliers included).")

    # --- Compute mean + SEM ---
    df_var['Distance (mm)'] = df_var['Distance (mm)'].astype(int)
    df_var['Repetitions'] = df_var['Repetitions'].astype(int)
    
    print(f"\nGrouping data for aggregation...")
    print(f"DataFrame shape before groupby: {df_var.shape}")
    print(f"DataFrame dtypes:\n{df_var.dtypes}")
    
    df_grouped = (
        df_var.groupby(['Distance (mm)', 'Method', 'Repetitions'], as_index=False, observed=True)
        .agg(mean_var=('Baseline Power (pA²/m⁴)', 'mean'),
             sem_var=('Baseline Power (pA²/m⁴)', lambda x: np.std(x, ddof=1)/np.sqrt(len(x)) if len(x) > 0 else 0))
    )
    
    df_grouped['Method'] = pd.Categorical(df_grouped['Method'], categories=method_order, ordered=True)
    
    print(f"Grouped data shape: {df_grouped.shape}")

    # --- Wilcoxon tests (paired by subject) ---
    sig_results = []
    
    n_subjects = df_var['Subject'].nunique()
    print(f"\nNumber of unique subjects: {n_subjects}")
    
    if n_subjects < 2:
        print("WARNING: Only one subject found. Wilcoxon test requires multiple subjects.")
        print("Skipping statistical testing.")
    else:
        for dist in dists:
            for method in method_order:
                method_data = df_var[(df_var['Distance (mm)'] == dist) &
                                     (df_var['Method'] == method)]
    
                for i in range(len(reps) - 1):
                    rep1, rep2 = reps[i], reps[i + 1]
    
                    # Collect paired data per subject
                    paired_data = []
                    for subj in subjects:
                        subj_data = method_data[method_data['Subject'] == subj]
                        val1 = subj_data.loc[subj_data['Repetitions'] == rep1, 'Baseline Power (pA²/m⁴)']
                        val2 = subj_data.loc[subj_data['Repetitions'] == rep2, 'Baseline Power (pA²/m⁴)']
                        if len(val1) > 0 and len(val2) > 0:
                            paired_data.append((val1.values[0], val2.values[0]))
    
                    if len(paired_data) < 2:
                        pval = np.nan
                    else:
                        vals1, vals2 = zip(*paired_data)
                        try:
                            stat, pval = wilcoxon(vals1, vals2, alternative='greater')
                        except ValueError:
                            pval = np.nan
    
                    sig_results.append({
                        'Distance (mm)': dist,
                        'Method': method,
                        'Rep1': rep1,
                        'Rep2': rep2,
                        'n_subjects_used': len(paired_data),
                        'pval': pval
                    })
    
    df_sig = pd.DataFrame(sig_results)
    
    if len(df_sig) > 0 and 'pval' in df_sig.columns:
        pvals = df_sig['pval'].dropna()
        if len(pvals) > 0:
            rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
            df_sig.loc[df_sig['pval'].notna(), 'pval_fdr'] = pvals_corrected
            df_sig.loc[df_sig['pval'].notna(), 'significant_fdr'] = rejected
    else:
        print("No statistical tests were performed (need multiple subjects).")


    # --- Plot setup ---
    font_scale = 1.6
    with sns.plotting_context("notebook", font_scale=font_scale):
    
        n_cols = 2 if len(dists) > 1 else 1
        n_rows = int(np.ceil(len(dists) / n_cols))
    
        fig2, axes2 = plt.subplots(
            n_rows, n_cols,
            figsize=(7 * n_cols, 6 * n_rows),
            sharey=False,
            squeeze=False
        )
    
        for idx, dist in enumerate(dists):
            row_idx, col_idx = divmod(idx, n_cols)
            ax = axes2[row_idx, col_idx]
    
            dist_data = df_grouped[df_grouped['Distance (mm)'] == dist].copy()
            dist_data = dist_data.sort_values('Method')
            ymax_local = 1.2 * dist_data["mean_var"].max() if not dist_data.empty else 1
    
            # --- Plot mean ± SEM ---
            for i, rep in enumerate(reps):
                rep_data = dist_data[dist_data['Repetitions'] == rep].copy()
                rep_data['method_idx'] = rep_data['Method'].map({m: idx for idx, m in enumerate(method_order)})
                rep_data = rep_data.sort_values('method_idx')
    
                ax.errorbar(
                    rep_data['method_idx'], rep_data['mean_var'],
                    yerr=rep_data['sem_var'],
                    fmt='-D', markersize=10, linewidth=2.5, capsize=5,
                    color=gray_colors[i], linestyle=line_styles[i],
                    label=f'{rep} trials'
                )
    
            # --- Log scale ---
            ax.set_yscale('log')
            ax.set_xticks(range(len(method_order)))
            ax.set_xticklabels(method_order)
    
            # --- Annotate significance ---
            if len(df_sig) > 0 and 'pval' in df_sig.columns:
                sig_dist = df_sig[df_sig['Distance (mm)'] == dist]
                ymax_local = ax.get_ylim()[1]
    
                for method in method_order:
                    method_sig = sig_dist[sig_dist['Method'] == method]
                    method_means = dist_data[dist_data['Method'] == method]
                    if method_means.empty:
                        continue
    
                    base_y = method_means['mean_var'].max()
                    if method_sig.empty:
                        continue
    
                    offset = -0.1
                    base_offset = 0.25 * ymax_local
                    step_offset = 0.15 * ymax_local
                    x_pos = method_order.index(method)
    
                    for j, row in enumerate(method_sig.itertuples()):
                        pval = getattr(row, 'pval_fdr', row.pval)
                        rep2 = row.Rep2
    
                        if pd.isna(pval):
                            continue
    
                        is_sig = getattr(row, 'significant_fdr', (pval < 0.05))
                        if is_sig:
                            if pval < 0.001:
                                mark = '***'
                            elif pval < 0.01:
                                mark = '**'
                            else:
                                mark = '*'
                            color = gray_colors[reps.index(rep2)]
                            ax.text(
                                x_pos,
                                base_y + base_offset + offset * step_offset,
                                mark,
                                ha='center', va='bottom',
                                color=color,
                                fontsize=22, weight='bold',
                                clip_on=False
                            )
                            offset += 1
    
            # --- Axis styling ---
            # Hide only top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
            # Keep bottom and left spines visible for all subplots
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
    
            # Uniform black spine styling
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('black')
    
            # Tick styling
            ax.tick_params(axis='x', rotation=0, length=5, width=1.2, color='black', labelcolor='black')
            ax.tick_params(axis='y', length=5, width=1.2, color='black', labelcolor='black')
    
            # --- Labels & titles ---
            ax.set_xlabel("Method", weight='bold')
            ax.set_ylabel("Baseline Power log(pA²/m⁴)", weight='bold')
            ax.set_title(f"{dist} mm", weight='bold')
    
            # Clean background (no grid)
            ax.grid(False)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
    
            # For log scale, set lower bound > 0
            if not dist_data.empty:
                ax.set_ylim(bottom=dist_data['mean_var'].min() * 0.5)
    
        # Turn off any unused subplots
        for idx in range(len(dists), n_rows * n_cols):
            axes2.flatten()[idx].axis('off')
    
        # --- Legend on right side ---
        handles, labels = ax.get_legend_handles_labels()
        fig2.legend(
            handles, labels,
            title='Number of trials',
            fontsize=18, title_fontsize=18,
            loc='center left', frameon=False,
            bbox_to_anchor=(1.02, 0.5)
        )
    
        plt.subplots_adjust(right=0.82, wspace=0.3, hspace=0.3)
        plt.show()



    # ------------------ Figure 3: Total Power vs Trial Number (using each subject's highest SNR source) ------------------
    print("\nGenerating Figure 3: Total Power vs Trial Number")
    
    power_data = []
    for subj in subjects:
        subj_source_idx = highest_snr_sources[subj]
        
        # Calculate subject-specific source-based scaling
        subj_source_scaling = {
            dist: {
                method: (
                    np.max(np.abs(stcs[subj][dist]["MNE"][reference_rep_idx].data[subj_source_idx]))
                    / np.max(np.abs(stcs[subj][dist][method][reference_rep_idx].data[subj_source_idx]))
                )
                for method in ["dSPM", "sLORETA", "eLORETA"]
            }
            for dist in dists
        }
        
        for dist in dists:
            for method in methods:
                for i, stc in enumerate(stcs[subj][dist][method]):
                    source_ts = stc.data[subj_source_idx]
                    stc_times = stc.times
                    if method == "MNE":
                        scaled_ts = source_ts * 1e12
                    else:
                        scaled_ts = source_ts * subj_source_scaling[dist][method] * 1e12
                    time_mask = (stc_times >= -0.1) & (stc_times <= 0.2)
                    total_power = np.sum(scaled_ts[time_mask] ** 2)
                    power_data.append({
                        'Subject': subj,
                        'Distance (mm)': dist,
                        'Method': method,
                        'Repetitions': reps[i],
                        'Total Power': total_power
                    })
    
    df_power = pd.DataFrame(power_data)
    
    stats = (
        df_power.groupby(['Distance (mm)', 'Method', 'Repetitions'], as_index=False)
        .agg(mean_power=('Total Power', 'mean'),
             sem_power=('Total Power', lambda x: np.std(x, ddof=1)/np.sqrt(len(x)) if len(x) > 0 else 0))
    )

    font_scale = 1.6
    # --- Plot configuration ---
    with sns.plotting_context("notebook", font_scale=font_scale):
        for i_dist, dist in enumerate(dists):
            row_data = stats[stats['Distance (mm)'] == dist].copy()
            
            # Calculate global min/max across all methods for this distance
            global_min = (row_data['mean_power'] - row_data['sem_power']).min()
            global_max = (row_data['mean_power'] + row_data['sem_power']).max()
            
            overall_max = row_data['mean_power'].max()
            scale_exp = 3 * (int(np.floor(np.log10(overall_max))) // 3)
            scale_factor = 10 ** scale_exp
            row_data['mean_power_scaled'] = row_data['mean_power'] / scale_factor
            row_data['sem_power_scaled'] = row_data['sem_power'] / scale_factor
    
            # 2x2 grid for methods
            n_rows, n_cols = 2, 2
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(12, 9),
                sharex=True, sharey=False
            )
            axes = axes.flatten()
    
            for j_method, method in enumerate(methods):
                ax = axes[j_method]
                method_data = row_data[row_data['Method'] == method].sort_values('Repetitions')
    
                # --- Plot mean ± SEM ---
                for k, rep in enumerate(reps):
                    rep_data = method_data[method_data['Repetitions'] == rep]
                    if not rep_data.empty:
                        ax.errorbar(
                            rep_data['Repetitions'], rep_data['mean_power_scaled'],
                            yerr=rep_data['sem_power_scaled'],
                            fmt='D', markersize=8, linewidth=2.5, capsize=4,
                            color=gray_colors[k], linestyle=line_styles[k],
                            label=f'{rep} trials',
                            elinewidth=2.5
                        )
    
                # --- Titles & labels ---
                ax.set_title(method, weight='bold')
    
                # Label y-axis only for first column
                if j_method % n_cols == 0:
                    ax.set_ylabel('Total Power (pA²/m⁴)', weight='bold')
                else:
                    ax.set_ylabel('')
    
                # Label x-axis for all plots (as requested)
                ax.set_xlabel('Number of trials')
    
                # --- Axis formatting ---
                row_idx, col_idx = divmod(j_method, n_cols)
    
                # Hide top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
    
                # ✅ Keep bottom spine visible for all
                ax.spines['bottom'].set_visible(True)
    
                # Only show left spine on first column
                if col_idx == 0:
                    ax.spines['left'].set_visible(True)
                else:
                    ax.spines['left'].set_visible(False)
                    ax.yaxis.set_visible(False)
    
                # Make visible spines consistent and clean
                for spine in ax.spines.values():
                    spine.set_color('black')
                    spine.set_linewidth(1.2)
    
                # Ticks and text
                ax.tick_params(axis='both', length=5, width=1.2, color='black')
                ax.tick_params(labelcolor='black')
    
                # Scale text in upper-left corner
                ax.text(
                    0.02, 0.99, f"×10$^{{{scale_exp}}}$",
                    transform=ax.transAxes,
                    ha='left', va='top',
                    fontsize=12 * font_scale
                )
    
                # Clean style (no grid)
                ax.grid(False)
                ax.xaxis.grid(False)
                ax.yaxis.grid(False)
    
            # --- Apply global y-limits ---
            global_min_scaled = global_min / scale_factor
            global_max_scaled = global_max / scale_factor
            for j_method in range(len(methods)):
                axes[j_method].set_ylim(global_min_scaled * 0.95, global_max_scaled * 1.05)
    
            # Turn off unused axes
            for ax in axes[len(methods):]:
                ax.axis('off')
    
            # --- Legend outside on the right ---
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles, labels,
                title='Number of trials',
                fontsize=12 * font_scale,
                title_fontsize=14 * font_scale,
                loc='center left', frameon=False,
                bbox_to_anchor=(1.02, 0.5)
            )
    
            plt.subplots_adjust(right=0.82, wspace=0.3, hspace=0.4)
            plt.show()




     
#-------------------------------------------------------------------------------------#
def plot_splits_and_nmse(p, method='eLORETA'):

    cwd = getcwd()
    reps = p.reps_split
    dist = 10  # Fixed distance

    # Load STCs for both splits
    split_stcs = {1: [], 2: []}
    for split in [1, 2]:
        for rep in reps:
            fname = f"current{rep}reps_{dist}mm_{method}_split{split}-lh.stc"
            stc_path = path.join(cwd, p.data, p.processed_data, 
                                 p.protocol, p.subj, p.session, 
                                 p.current_dir, fname)
            split_stcs[split].append(mne.read_source_estimate(stc_path))

    # Time vector (assume same for all)
    times = split_stcs[1][0].times

    # Get max-SNR source index for each split (based on largest rep)
    ref_idx = -1  # last in list = highest rep
    max_source_idx = {
        split: np.argmax(np.max(np.abs(split_stcs[split][ref_idx].data), axis=1))
        for split in [1, 2]
    }
    
    #subj02
    #max_source_idx = { 1:544, 2:544}
    #max_source_idx = { 1:923, 2:923}

    #subj03
    #max_source_idx = { 1:1124, 2:1124}

    # subj04
    #max_source_idx = { 1:1749, 2:1749}

    # subj05
    #max_source_idx = { 1:1384, 2:1384}
    
    # Prepare DataFrame for split plots (each plot uses its own max source)
    df_list = []
    for split in [1, 2]:
        idx = max_source_idx[split]
        for i, stc in enumerate(split_stcs[split]):
            scaled_activity = stc.data[idx] * 1e12  # Convert to pA
            df_temp = pd.DataFrame({
                "Time (s)": times,
                "Activity (pA)": scaled_activity,
                "Repetitions": reps[i],
                "Split": f"Split {split}"
            })
            df_list.append(df_temp)

    df = pd.concat(df_list)

    # --- Plot Source Activity for Each Split ---
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for i, split in enumerate([1, 2]):
        sns.lineplot(
            data=df[df["Split"] == f"Split {split}"],
            x="Time (s)", y="Activity (pA)", hue="Repetitions", ax=axes[i])
        axes[i].set_title(f"{method} - Split {split} (Source {max_source_idx[split]})", fontsize=16)
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Source Activity (pA/m²)")
        axes[i].legend(title="Reps")

    plt.suptitle(f"Source Currents at Max-SNR Sources (Split-Specific)", fontsize=20)
    plt.tight_layout()
    plt.show()

    # --- NMSE between split max sources ---
    nmse_vals = []
    for stc1, stc2 in zip(split_stcs[1], split_stcs[2]):
        s1 = stc1.data[max_source_idx[1]]
        s2 = stc2.data[max_source_idx[2]]
        nmse = np.mean((s1 - s2) ** 2) / (np.mean(s1 ** 2) + 1e-12)  # add epsilon to avoid div-by-zero
        nmse_vals.append(nmse)

    # --- Plot NMSE across reps ---
    plt.figure(figsize=(10, 5))
    sns.barplot(x=reps, y=nmse_vals, palette='mako')
    plt.title(f"NMSE Between Split-Specific Max-SNR Sources ({method})", fontsize=18)
    plt.xlabel("Number of Repetitions")
    plt.ylabel("Normalized MSE")
    plt.tight_layout()
    plt.show()

#---------------------------------------------------------------------------------------#
def plot_nmse_across_dists(p, method='dSPM'):

    cwd = getcwd()
    reps = p.reps_split
    dists = p.dists  # List of distances
    nmse_records = []

    for dist in dists:
        split_stcs = {1: [], 2: []}
        for split in [1, 2]:
            for rep in reps:
                fname = f"current{rep}reps_{dist}mm_{method}_split{split}-lh.stc"
                stc_path = path.join(cwd, p.data, p.processed_data, 
                                     p.protocol, p.subj, p.session, 
                                     p.current_dir, fname)
                try:
                    stc = mne.read_source_estimate(stc_path)
                    split_stcs[split].append(stc)
                except FileNotFoundError:
                    print(f"Missing STC file: {stc_path}")
                    continue

        if not split_stcs[1] or not split_stcs[2]:
            print(f"Skipping dist {dist} due to missing data.")
            continue

        # Get max-SNR source index for each split (from last rep, assumed highest SNR)
        max_source_idx = {
            split: np.argmax(np.max(np.abs(split_stcs[split][-1].data), axis=1))
            for split in [1, 2]
        }

        # subj04
        max_source_idx = { 1:max_source_idx[1], 2:max_source_idx[1]}
    
        # Compute NMSE using each split’s best source
        for i, (stc1, stc2) in enumerate(zip(split_stcs[1], split_stcs[2])):
            s1 = stc1.data[max_source_idx[1]]
            s2 = stc2.data[max_source_idx[2]]
            nmse = np.mean((s1 - s2) ** 2) / (np.mean(s1 ** 2) + 1e-12)
            nmse_records.append({
                "Repetitions": reps[i],
                "NMSE": nmse,
                "Distance (mm)": dist
            })

    # Create DataFrame
    df = pd.DataFrame(nmse_records)

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Repetitions", y="NMSE", hue="Distance (mm)", marker="o")
    plt.title(f"NMSE Between Split-Specific Max Sources ({method})", fontsize=18)
    plt.xlabel("Number of Repetitions", fontsize=20)
    plt.ylabel("Normalized Mean Squared Error", fontsize=20)
    plt.legend(title="Dipole Distance", fontsize=20)
    plt.tight_layout()
    plt.show()
#---------------------------------------------------------------------------------------#

def plot_split_nmse_participants(p, method='eLORETA'):
    cwd = getcwd()
    reps = p.reps_split
    dist = 10  # Fixed distance
    participant_list = p.subjects
    nmse_records = []

    for subj in participant_list:
        split_stcs = {1: [], 2: []}
        for split in [1, 2]:
            for rep in reps:
                fname = f"current{rep}reps_{dist}mm_{method}_split{split}-lh.stc"
                stc_path = path.join(cwd, p.data, p.processed_data,
                                     p.protocol, subj, p.session,
                                     p.current_dir, fname)
                try:
                    stc = mne.read_source_estimate(stc_path)
                    split_stcs[split].append(stc)
                except FileNotFoundError:
                    print(f"Missing STC file: {stc_path}")
                    continue

        if not split_stcs[1] or not split_stcs[2]:
            print(f"Skipping subject {subj} due to missing data.")
            continue

        # Max-SNR source index for each split (based on largest rep)
        max_source_idx = {
            split: np.argmax(np.max(np.abs(split_stcs[split][-1].data), axis=1))
            for split in [1, 2]
        }

        # subj04 override (optional, maybe for debugging?)
        max_source_idx = {1: 1749, 2: 1749}

        # NMSE across reps (split-specific max sources)
        for i, (stc1, stc2) in enumerate(zip(split_stcs[1], split_stcs[2])):
            s1 = stc1.data[max_source_idx[1]]
            s2 = stc2.data[max_source_idx[2]]
            nmse = np.mean((s1 - s2) ** 2) / (np.mean(s1 ** 2) + 1e-12)
            nmse_records.append({
                "Subject": subj,
                "Repetitions": reps[i],
                "NMSE": nmse
            })

    # --- DataFrames ---
    df = pd.DataFrame(nmse_records)
    df_avg = df.groupby("Repetitions", as_index=False)["NMSE"].mean()
    df_avg["Subject"] = "Average"

    # --- Plot ---
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Define a consistent palette
    subjects = sorted(df["Subject"].unique())
    palette = dict(zip(subjects, sns.color_palette("colorblind", n_colors=len(subjects))))

    # Plot individual participants
    sns.lineplot(
    data=df,
    x="Repetitions", y="NMSE", hue="Subject",
    palette=palette,
    linewidth=1.2, alpha=0.6, marker='o', markersize=5,
    legend=False  
    )

    # Plot average with bold line and standout color
    sns.lineplot(
        data=df_avg,
        x="Repetitions", y="NMSE", label="Average",
        color="black", linewidth=3.5, marker='D', markersize=8
    )

    # Styling
    plt.title(f"NMSE Between Split-Specific Max Sources\n({method}, {dist} mm)", fontsize=18, weight='bold')
    plt.xlabel("Number of Repetitions", fontsize=20)
    plt.ylabel("Normalized Mean Squared Error", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Custom legend labels
    custom_labels = {
        "01": "01",
        "02": "02",
        "03": "03",
        "04": "04",
    }
    
    # Manually add each subject line to legend
    from matplotlib.lines import Line2D
    
    # Build legend elements for each subject
    legend_elements = [
        Line2D(
            [0], [0],
            color=palette[subj],
            lw=2,
            marker='o',
            markersize=14,
            label=custom_labels.get(subj, subj)
        )
        for subj in custom_labels
    ]
    
    # Add the "Average" line manually
    legend_elements.append(
        Line2D(
            [0], [0],
            color='black',
            lw=3,
            marker='D',
            markersize=14,
            label='Average'
        )
    )
    
    plt.legend(handles=legend_elements, title="Subject", fontsize=12, frameon=True)
    plt.tight_layout()
    plt.show()

