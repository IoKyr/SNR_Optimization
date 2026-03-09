from os import getcwd
from os import path
import os
import numpy as np
import mne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from itertools import product
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from scipy.stats import entropy as shannon_entropy
import itertools
import re
import statsmodels.formula.api as smf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse, Polygon
from scipy.interpolate import griddata
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from scipy.stats import kstest, shapiro, anderson, levene, norm, wilcoxon, kruskal, mannwhitneyu, spearmanr
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D
from mne import Label
from statsmodels.stats.multitest import fdrcorrection



def snr_hammerer(p,data, times):             
    t_min, t_max = p.tmin_snr, p.tmax_snr
    t_min_base, t_max_base = p.tmin_base_snr, p.tmax_base_snr

    # Find indices of the time window
    idx_min = np.argmin(np.abs(times - t_min))
    idx_max = np.argmin(np.abs(times - t_max))
    
    # Find indices of the baseline
    idx_min_base = np.argmin(np.abs(times - t_min_base))
    idx_max_base = np.argmin(np.abs(times - t_max_base))

    # Compute mean amplitude in the window (across time)
    mean_amplitude = np.mean(data[:, idx_min:idx_max]** 2, axis=1)  # Shape: (n_channels,)
    std_amplitude  = np.std(data[:, idx_min_base:idx_max_base], axis=1)
    std_amplitude[std_amplitude == 0] = 1e-10  # Prevent division by zero
    snr = np.sqrt(mean_amplitude) / std_amplitude
    
    return snr
    
def measure_snr_el(p, output_csv="snr_01.csv"):
    print("Measure SNR on electrode level: started")
    cwd = getcwd()
    processedeegpath = path.join(cwd, p.data, p.processed_data, 
                                 p.protocol, p.subj, p.session, 
                                 p.eeg_dir)
    
    fullpath_epochs = path.join(processedeegpath, p.epochsfile)
    epochs = mne.read_epochs(fullpath_epochs)
    selected_epochs = epochs[p.selected_epoch]

    n_epochs = len(selected_epochs)
    n_select = [125, 250, 500, 1000, 1500, 2000]
    n_repeats = 20

    data_list = []
    np.random.seed(11)

    for n in n_select:
        snr_values_per_channel = {ch: [] for ch in selected_epochs.ch_names}
        signal_power_per_channel = {ch: [] for ch in selected_epochs.ch_names}
        noise_power_per_channel = {ch: [] for ch in selected_epochs.ch_names}
        
        for _ in range(n_repeats):
            random_indices = np.random.choice(n_epochs, n, replace=False)
            random_subset = selected_epochs[random_indices]

            erp = random_subset.average()
            snr = snr_hammerer(p, erp.data, erp.times)

            # Compute power values (copied logic from snr_hammerer)
            t_min, t_max = p.tmin_snr, p.tmax_snr
            t_min_base, t_max_base = p.tmin_base_snr, p.tmax_base_snr
            idx_min = np.argmin(np.abs(erp.times - t_min))
            idx_max = np.argmin(np.abs(erp.times - t_max))
            idx_min_base = np.argmin(np.abs(erp.times - t_min_base))
            idx_max_base = np.argmin(np.abs(erp.times - t_max_base))

            signal_power = np.mean(erp.data[:, idx_min:idx_max] ** 2, axis=1)
            noise_std = np.std(erp.data[:, idx_min_base:idx_max_base], axis=1)
            noise_std[noise_std == 0] = 1e-10
            noise_power = noise_std ** 2

            for ch_idx, ch_name in enumerate(selected_epochs.ch_names):
                snr_values_per_channel[ch_name].append(snr[ch_idx])
                signal_power_per_channel[ch_name].append(signal_power[ch_idx])
                noise_power_per_channel[ch_name].append(noise_power[ch_idx])

        for ch_name in selected_epochs.ch_names:
            data_list.append({
                "n_select": n,
                "channel": ch_name,
                "snr_mean": np.mean(snr_values_per_channel[ch_name]),
                "snr_std": np.std(snr_values_per_channel[ch_name]),
                "signal_power": np.mean(signal_power_per_channel[ch_name]),
                "noise_power": np.mean(noise_power_per_channel[ch_name])
            })

    df = pd.DataFrame(data_list)

    output_path = path.join(cwd, p.data, p.processed_data, 
                            p.protocol, p.subj, p.session, 
                            p.current_dir, output_csv)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print("Measure SNR on electrode level: completed")


#------------------------------------------------------------------------------------#
def plot_snr_el_multi_subject(p):
    print("Plotting SNR, signal power, and noise power: started")
    cwd = getcwd()
    subjects = p.subjects
    channels_of_interest = ["CP3", "CPz", "CP4"]
    all_data = []
    all_data_full = []  # For topoplots with all channels
    
    for subj in subjects:
        subj_number = ''.join(filter(str.isdigit, subj))
        filename = f"snr_{subj_number}_el.csv"
        filepath = path.join(cwd, p.data, p.processed_data, 
                             p.protocol, subj, p.session, 
                             p.current_dir, filename)
        if not path.exists(filepath):
            print(f"Missing SNR file for {subj}. Skipping.")
            continue
        
        # Load full data for topoplots
        df_full = pd.read_csv(filepath)
        df_full["Subject"] = subj
        all_data_full.append(df_full)
        
        # Load filtered data for later analysis
        df = df_full[df_full["channel"].isin(channels_of_interest)].copy()
        all_data.append(df)
    
    if not all_data:
        print("No valid SNR data loaded. Exiting.")
        return
    
    df_all = pd.concat(all_data)
    df_all_full = pd.concat(all_data_full)  # All channels for topoplots
    
    # --- Print SNR Summary Table ---
    print("\n" + "="*90)
    print("SNR SUMMARY TABLE (Mean ± SD) - By Number of Trials")
    print("="*90)
    
    # Calculate mean and std for each channel and repetition number
    snr_summary = df_all.groupby(['n_select', 'channel'])['snr_mean'].agg(['mean', 'std']).round(2)
    
    # Get unique repetition numbers and sort them
    repetition_numbers = sorted(df_all['n_select'].unique())
    
    #--------------------------------------------------------------------------------#
    # Calculate SNR gains   
    snr_gains = {}
    efficiency_points = {}
    
    for channel in channels_of_interest:
        channel_data = []
        for n_rep in repetition_numbers:
            if (n_rep, channel) in snr_summary.index:
                mean_val = snr_summary.loc[(n_rep, channel), 'mean']
                channel_data.append((n_rep, mean_val))
        
        if len(channel_data) >= 3:  # Need at least 3 points for meaningful analysis
            # Calculate gains (improvement from previous repetition)
            gains = []
            efficiencies = []  # SNR gain per repetition added
            
            for i in range(1, len(channel_data)):
                gain = channel_data[i][1] - channel_data[i-1][1]
                reps_added = channel_data[i][0] - channel_data[i-1][0]
                efficiency = gain / reps_added if reps_added > 0 else 0
                gains.append((channel_data[i][0], gain))
                efficiencies.append((channel_data[i][0], efficiency))
                    
            # Method 2: Efficiency threshold (when efficiency drops below 50% of max)
            if efficiencies:
                max_efficiency = max(eff[1] for eff in efficiencies)
                eff_threshold = max_efficiency * 0.5
                
                efficiency_point = None
                for rep, eff in efficiencies:
                    if eff < eff_threshold:
                        efficiency_point = rep
                        break
                efficiency_points[channel] = efficiency_point
            
            snr_gains[channel] = gains
    
    # Print header
    header = f"{'Trials':<12}"
    for channel in channels_of_interest:
        header += f"{channel + ' (Mean±SD)':<20}{'Gain':<10}{'Efficiency':<12}"
    print(header)
    print("-" * 120)
    
    # Print data for each repetition number
    for i, n_rep in enumerate(repetition_numbers):
        row = f"{n_rep:<12}"
        for channel in channels_of_interest:
            if (n_rep, channel) in snr_summary.index:
                mean_val = snr_summary.loc[(n_rep, channel), 'mean']
                std_val = snr_summary.loc[(n_rep, channel), 'std']
                
                # Find gain and efficiency for this repetition
                gain_str = ""
                eff_str = ""
                if channel in snr_gains:
                    for rep_gain, gain_val in snr_gains[channel]:
                        if rep_gain == n_rep:
                            gain_str = f"+{gain_val:.2f}"
                            break
                    # Find efficiency
                    prev_rep = repetition_numbers[i-1] if i > 0 else None
                    if prev_rep is not None:
                        reps_added = n_rep - prev_rep
                        if gain_str:
                            gain_val = float(gain_str[1:])  # Remove +
                            efficiency = gain_val / reps_added
                            eff_str = f"{efficiency:.3f}"
                
                snr_str = f"{mean_val:.2f}±{std_val:.2f}"
                row += f"{snr_str:<20}{gain_str:<10}{eff_str:<12}"
            else:
                row += f"{'No data':<20}{'--':<10}{'--':<12}"
        
        # Add markers at end of row
        markers = []
        for channel in channels_of_interest:
            if channel in efficiency_points and efficiency_points[channel] == n_rep:
                markers.append("LOW-EFF")
        
        if markers:
            row += f" ← {'/'.join(markers)}"
        
        print(row)
    
    print("="*120)
    print("Note: 'LOW-EFF' = efficiency <50% of max")
    print("      'Efficiency' = SNR gain per repetition added")
    print("="*120 + "\n")
    
    # --- Statistical Analysis Section ---
    from scipy.stats import friedmanchisquare
    from matplotlib.colors import TwoSlopeNorm
    
    # Dictionary to map metric names to column names
    metric_mapping = {
        "SNR": "snr_mean",
        "Signal Power": "signal_power", 
        "Noise Power": "noise_power"
    }
    
    # Store significance results for plotting
    significance_results = {}
    
    for metric_name, metric_column in metric_mapping.items():
        print(f"\nNon-parametric statistics on {metric_name} (per channel):")
        print("=" * 80)
        
        significance_results[metric_name] = {}
        
        for ch in channels_of_interest:
            print(f"\n→ Channel: {ch}")
            
            df_ch = df_all[df_all["channel"] == ch].copy()
            
            # Use the metric as is (you can add log transformation if needed)
            df_ch["metric_test"] = df_ch[metric_column]
            
            pivot_df = df_ch.pivot(index="Subject", columns="n_select", values="metric_test")
            
            # Drop subjects with missing values across reps
            pivot_df = pivot_df.dropna()
            reps_available = sorted(pivot_df.columns.tolist())
            
            if len(pivot_df) == 0:
                print("Not enough subjects with complete data. Skipping.")
                continue
            
            # Friedman test across all repetition levels
            stat, p = friedmanchisquare(*[pivot_df[col] for col in reps_available])
            print(f"Friedman χ²={stat:.3f}, p={p:.4e}")
            
            # Initialize significance tracking for this metric and channel
            significance_results[metric_name][ch] = {}
            
            if p < 2.0:
                print("↳ Significant overall effect → running pairwise Wilcoxon with FDR correction:")
                
                # Collect Wilcoxon results
                p_values = []
                comparisons = []
                for rep1, rep2 in itertools.combinations(reps_available, 2):
                    try:
                        stat, pval = wilcoxon(pivot_df[rep1], pivot_df[rep2])
                        p_values.append(pval)
                        comparisons.append((rep1, rep2))
                    except ValueError:
                        p_values.append(np.nan)
                        comparisons.append((rep1, rep2))
                
                # Multiple comparison correction
                reject, pvals_corr, _, _ = multipletests(p_values, method="fdr_bh")
                
                # Store significance results for consecutive comparisons
                for (rep1, rep2), p_corr, rej in zip(comparisons, pvals_corr, reject):
                    if not np.isnan(p_corr):
                        significance_results[metric_name][ch][(rep1, rep2)] = rej
                
                # Build corrected p-value matrix
                pval_matrix_corr = pd.DataFrame(
                    np.ones((len(reps_available), len(reps_available))),
                    index=reps_available, columns=reps_available
                )
                for (rep1, rep2), p_corr in zip(comparisons, pvals_corr):
                    pval_matrix_corr.loc[rep1, rep2] = p_corr
                    pval_matrix_corr.loc[rep2, rep1] = p_corr
                
                np.fill_diagonal(pval_matrix_corr.values, 1.0)
                
                # Print table of results
                print(f"{'Comparison':<15}{'Corrected p':<15}{'Significant':<12}")
                print("-" * 45)
                for (rep1, rep2), p_corr, rej in zip(comparisons, pvals_corr, reject):
                    if np.isnan(p_corr):
                        continue
                    print(f"{rep1} vs {rep2:<7}{p_corr:<15.4f}{str(rej):<12}")
                
                # Store p-value matrix for 3x3 plot
                significance_results[metric_name][ch]['pval_matrix'] = pval_matrix_corr
                significance_results[metric_name][ch]['reps_available'] = reps_available
            else:
                print("↳ No significant effect across repetitions.")
                # Still store empty results for consistency
                significance_results[metric_name][ch]['pval_matrix'] = None
                significance_results[metric_name][ch]['reps_available'] = []
    
    # --- Create 3x3 Wilcoxon Heatmap Figure ---
    fig_stats, axes_stats = plt.subplots(3, 3, figsize=(18, 12))

    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    
    metric_names = ["SNR", "Signal Power", "Noise Power"]
    
    for row_idx, metric_name in enumerate(metric_names):
        for col_idx, ch in enumerate(channels_of_interest):
            ax = axes_stats[row_idx, col_idx]
            
            if (metric_name in significance_results and 
                ch in significance_results[metric_name] and
                significance_results[metric_name][ch].get('pval_matrix') is not None):
                
                pval_matrix_corr = significance_results[metric_name][ch]['pval_matrix']
                
                # Create mask for upper triangle
                mask = np.triu(np.ones_like(pval_matrix_corr, dtype=bool))
                significance_level = 0.05
                
                # Diverging colormap centered on significance threshold (0.05)
                cmap = plt.cm.RdBu_r
                norm = TwoSlopeNorm(vmin=0, vcenter=significance_level, vmax=1)
                
                sns.heatmap(
                    pval_matrix_corr,
                    mask=mask,
                    annot=True,
                    fmt=".3f",
                    cmap=cmap,
                    norm=norm,
                    cbar=False,  # No individual colorbars
                    linewidths=0.5,
                    linecolor='gray',
                    annot_kws={"size": 10},
                    square=True,
                    ax=ax
                )
                
            else:
                # No significant data - create empty plot
                ax.text(0.5, 0.5, 'No significant\neffect', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Set titles and labels
            if row_idx == 0:
                ax.set_title(f"{ch}", fontsize=16, weight="bold")
            if col_idx == 0:
                ylabel = "Number of trials"
                ax.set_ylabel(ylabel, fontsize=14)
            if row_idx == 2:
                ax.set_xlabel("Number of trials", fontsize=14)
            
            # Adjust tick label sizes
            ax.tick_params(labelsize=10)
    
    plt.suptitle("Wilcoxon p-values (FDR corrected)", fontsize=16, weight="bold")
    plt.tight_layout()
    
    # Add a single colorbar at the very right of the figure
    from matplotlib.colors import TwoSlopeNorm
    significance_level = 0.05
    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(vmin=0, vcenter=significance_level, vmax=1)
    
    # Create a ScalarMappable for the colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Add colorbar with specific positioning
    cbar_ax = fig_stats.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig_stats.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Corrected p-value", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    plt.show()

    # --- Plot Setup ---
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
            'font.size': 20,        # Default text size
            'axes.labelsize': 20,   # Axis labels
            'axes.titlesize': 20,   # Subplot titles
            'xtick.labelsize': 20,  # X-axis tick labels
            'ytick.labelsize': 20,  # Y-axis tick labels
            'legend.fontsize': 18,  # Legend text
        })
    
    # Define metrics
    metrics = {
        "SNR": "snr_mean",
        "Signal RMS (fV)": "signal_power",
        "Baseline SD (fV)": "noise_power"
    }
    
    # Map back to original names for significance lookup
    metric_name_mapping = {
        "SNR": "SNR",
        "Signal RMS (fV)": "Signal Power",
        "Baseline SD (fV)": "Noise Power"
    }
    
    # Scaling factors for power metrics
    metric_scaling = {
        "SNR": 1,
        "Signal RMS (fV)": 1e12,
        "Baseline SD (fV)": 1e12
    }
    
    # Create separate figure for each metric with consistent layout
    for metric_title, metric in metrics.items():
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True, sharey=True)
        
        # Set consistent subplot positioning for alignment across figures
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.15, wspace=0.3)
        scaling_factor = metric_scaling[metric_title]
        
        # Store significance markers for later plotting
        significance_markers = []
        
        for col_idx, ch in enumerate(channels_of_interest):
            ax = axes[col_idx]
            df_ch = df_all[df_all["channel"] == ch].copy()
            
            # Apply scaling
            df_ch[metric] = df_ch[metric] * scaling_factor
            
            # Plot individual subjects in grey with triangular markers
            for subject in df_ch["Subject"].unique():
                subj_data = df_ch[df_ch["Subject"] == subject]
                ax.plot(
                    subj_data["n_select"],
                    subj_data[metric],
                    color="grey",
                    marker="^",
                    linewidth=1,
                    alpha=0.3,
                    markersize=4
                )
            
            # Plot mean ± std across subjects
            grouped = df_ch.groupby("n_select")[metric]
            mean_vals = grouped.mean()
            std_vals = grouped.std()
            reps_sorted = sorted(df_ch["n_select"].unique())
            
            ax.plot(
                reps_sorted,
                mean_vals[reps_sorted],
                color="black",
                marker="^",
                linewidth=2.5,
                markersize=8,
                label="Mean ± Std"
            )
            ax.fill_between(
                reps_sorted,
                mean_vals[reps_sorted] - std_vals[reps_sorted],
                mean_vals[reps_sorted] + std_vals[reps_sorted],
                color="grey",
                alpha=0.2
            )
            
            # Store significance info for later plotting
            metric_for_lookup = metric_name_mapping[metric_title]
            if (metric_for_lookup in significance_results and 
                ch in significance_results[metric_for_lookup] and
                significance_results[metric_for_lookup][ch].get('pval_matrix') is not None):
                
                pval_matrix = significance_results[metric_for_lookup][ch]['pval_matrix']
                trials = significance_results[metric_for_lookup][ch]['reps_available']
                y_vals = df_ch.groupby('n_select')[metric].mean()
                
                significance_markers.append({
                    'ax': ax,
                    'col_idx': col_idx,
                    'pval_matrix': pval_matrix,
                    'trials': trials,
                    'y_vals': y_vals,
                    'y_range': y_vals.max() - y_vals.min()
                })
            
            # Plot formatting
            ax.set_title(f"{ch}", fontsize=26, weight="bold")
            ax.set_xlabel("Number of trials")
            
            # Only label the y-axis on the leftmost subplot
            if col_idx == 0:
                ax.set_ylabel(metric_title)
            else:
                ax.set_ylabel("")
            
            # --- Aesthetic improvements ---
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            
            if col_idx == 0:
                ax.spines['left'].set_visible(True)
            else:
                ax.spines['left'].set_visible(False)
                    
            # Optional: make spines thinner and lighter
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('black')
            
            # Remove gridlines entirely
            ax.grid(False)
            ax.xaxis.grid(False, which='both') # ensure no x-grid
            ax.yaxis.grid(False, which='both') # ensure no y-grid
            
            # Only keep left y-axis ticks
            if col_idx != 0:
                ax.yaxis.set_visible(False)

            
            if col_idx == 0:
                ax.set_ylabel(metric_title)
            else:
                ax.set_ylabel("")
            
        # Set shared y-limits across all subplots
        y_mins, y_maxs = [], []
        for col_idx, ch in enumerate(channels_of_interest):
            df_ch = df_all[df_all["channel"] == ch]
            y_vals = df_ch[metric].values * scaling_factor
            y_mins.append(np.nanmin(y_vals))
            y_maxs.append(np.nanmax(y_vals))
        
        # Extend ylim for significance markers
        data_min, data_max = min(y_mins), max(y_maxs)
        shared_ylim = (data_min, data_max * 1.2)
        
        # Apply to all axes
        for ax in axes:
            ax.set_ylim(shared_ylim)
            ax.set_xlim((125, 2000))
        
        # Add significance markers AFTER ylim is set
        vertical_height_points = 6
        
        for marker_info in significance_markers:
            ax = marker_info['ax']
            pval_matrix = marker_info['pval_matrix']
            trials = marker_info['trials']
            
            ylim = ax.get_ylim()
            y_fixed = ylim[0] + 0.6 * (ylim[1] - ylim[0])
            
            # Consistent vertical tick size
            y_lim_range = ylim[1] - ylim[0]
            vertical_height_data = vertical_height_points * y_lim_range / (ax.bbox.height * 72/ax.figure.dpi)
            
            for trial1, trial2 in zip(trials[:-1], trials[1:]):
                if pval_matrix.loc[trial1, trial2] < 0.05:
                    x_pos = (trial1 + trial2) / 2
                    sig_color = "#d62728"  # matplotlib's default red
                    
                    # Horizontal line
                    ax.plot([trial1, trial2], [y_fixed, y_fixed], color=sig_color, linewidth=1.5)
                    
                    # Vertical lines
                    ax.plot([trial1, trial1], [y_fixed - vertical_height_data, y_fixed + vertical_height_data], 
                            color=sig_color, linewidth=2, antialiased=False)
                    ax.plot([trial2, trial2], [y_fixed - vertical_height_data, y_fixed + vertical_height_data], 
                            color=sig_color, linewidth=2, antialiased=False)
                    
                    # Asterisk
                    ax.text(x_pos, y_fixed, '*', fontsize=26,
                            ha='center', va='center', weight='bold', color=sig_color)
        
        # Add legend to the rightmost subplot
        custom_lines = [
            Line2D([0], [0], color='grey', lw=1.2, alpha=0.5, label='Individual Participants (N=10)'),
            Line2D([0], [0], color='black', lw=2.5, label='Group Mean ± Std'),
            Line2D([0], [0], color='red', lw=2, marker='*', markersize=10, label='Significant change (p<0.05)')
        ]
        axes[2].legend(handles=custom_lines, loc="upper right")

        
        #plt.subplots_adjust(wspace=0.1)
        plt.show()
    
    
    print("Plotting SNR, signal power, and noise power: completed")


    # Graphical Abstract Plot
    # --- Simplified SNR Plot (CP3 + CP4) ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    metric = "snr_mean"
    scaling_factor = 1
    
    # Define line styles and colors for each channel
    line_styles = {"CP3": "-", "CP4": "--"}
    grey_colors = {"CP3": "#666666", "CP4": "#b4cde0"}  # CP3 darker
    
    for ch in ["CP3", "CP4"]:
        df_ch = df_all[df_all["channel"] == ch].copy()
        df_ch[metric] = df_ch[metric] * scaling_factor
        
        # Plot individual subjects in grey with channel-specific line style and color
        for subject in df_ch["Subject"].unique():
            subj_data = df_ch[df_ch["Subject"] == subject]
            ax.plot(
                subj_data["n_select"],
                subj_data[metric],
                color=grey_colors[ch],
                marker="^",
                linestyle=line_styles[ch],
                linewidth=1,
                alpha=0.3,
                markersize=4
            )
        
        # Plot mean ± std
        grouped = df_ch.groupby("n_select")[metric]
        mean_vals = grouped.mean()
        std_vals = grouped.std()
        reps_sorted = sorted(df_ch["n_select"].unique())
        
        ax.plot(
            reps_sorted,
            mean_vals[reps_sorted],
            color="black",
            marker="^",
            linestyle=line_styles[ch],
            linewidth=2.5,
            markersize=8,
            label=ch
        )
        ax.fill_between(
            reps_sorted,
            mean_vals[reps_sorted] - std_vals[reps_sorted],
            mean_vals[reps_sorted] + std_vals[reps_sorted],
            color=grey_colors[ch],
            alpha=0.4
        )
    
    # Formatting
    ax.set_xlabel("Number of trials")
    ax.set_ylabel("SNR")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')
    
    ax.grid(False)
    ax.set_xlim((125, 2000))
    
    plt.tight_layout()
    plt.show()
    
    print("Simplified SNR plot (CP3 + CP4): completed")



def plot_snr_from_csv(p, csv_file="snr_01.csv", normalize="None", reference_channel=None, hist_n_select=None):
    cwd = getcwd()
    loadpath = path.join(cwd, p.data, p.processed_data, 
                         p.protocol, p.subj, p.session, 
                         p.current_dir, csv_file)
    df = pd.read_csv(loadpath)

    selected_electrodes = ['C5','C3','C1','Cz','C2','C4','C6']
    electrode_colors = {
        'C5': '#CC79A7',
        'C3': '#D55E00',
        'C6': '#0072B2',
        'Cz': '#F0E442',
        'C2': '#009E73',
        'C4': '#56B4E9',
        'C1': '#E69F00',
    }

    df_selected = df.query("channel in @selected_electrodes").copy()

    if normalize == "per_channel":
        df["snr_mean"] = df.groupby("channel")["snr_mean"].transform(lambda x: x / x.max())
        df["snr_std"] = df.groupby("channel")["snr_std"].transform(lambda x: x / x.max())
        df_selected = df.query("channel in @selected_electrodes").copy()

    elif normalize == "global" and reference_channel:
        ref_max = df[df["channel"] == reference_channel]["snr_mean"].max()
        if pd.notna(ref_max) and ref_max > 0:
            df["snr_mean"] = df["snr_mean"] / ref_max
            df["snr_std"] = df["snr_std"] / ref_max
            df_selected = df.query("channel in @selected_electrodes").copy()

    # === LINE PLOT FOR SELECTED ELECTRODES ===
    mpl.rcParams.update({'font.size': 20})
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_selected, 
                 x="n_select", 
                 y="snr_mean",
                 hue='channel', 
                 marker="o",
                 palette=electrode_colors)

    for electrode in selected_electrodes:
        df_electrode = df_selected[df_selected["channel"] == electrode]
        plt.fill_between(
            df_electrode["n_select"], 
            df_electrode["snr_mean"] - df_electrode["snr_std"], 
            df_electrode["snr_mean"] + df_electrode["snr_std"], 
            alpha=0.1,
            color=electrode_colors[electrode]
        )

    plt.xlabel("Number of Repetitions")
    plt.ylabel("Mean SNR")
    plt.title("Participant 1")

    handles, labels = plt.gca().get_legend_handles_labels()
    desired_order = ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6']
    sorted_handles_labels = [(h, l) for l, h in zip(labels, handles) if l in desired_order]
    sorted_handles_labels.sort(key=lambda x: desired_order.index(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    plt.legend(handles=sorted_handles, labels=sorted_labels, title="Electrode")
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === HISTOGRAM FOR ALL ELECTRODES, SPECIFIC n_select ===
    if hist_n_select is not None:
        df_hist = df[df["n_select"] == hist_n_select]
        if df_hist.empty:
            print(f"No data found for n_select = {hist_n_select}. Histogram skipped.")
        else:
            plt.figure(figsize=(8, 5))
            sns.set(style="whitegrid")
            sns.histplot(df_hist["snr_mean"], bins=30, kde=True, color="darkcyan", edgecolor="black")

            plt.xlabel("Mean SNR", fontsize=16)
            plt.ylabel("Number of Electrodes", fontsize=16)
            plt.title(f"Histogram of Mean SNR Across All Electrodes\n(n_select = {hist_n_select})", fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(True)
            plt.tight_layout()
            plt.show()


            plt.figure(figsize=(10, 6))
            sns.boxplot(x="n_select", y="snr_mean", data=df[df["n_select"] == hist_n_select], color="lightgray")
            sns.swarmplot(x="n_select", y="snr_mean", data=df[df["n_select"] == hist_n_select], color="teal", alpha=0.7)
            
            plt.ylabel("Mean SNR")
            plt.xlabel("")
            plt.title(f"SNR Distribution Across All Electrodes (n_select = {hist_n_select})")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

#------------------------------------------------------------------------------------#    
def measure_snr_src_repnum_dist(p, method, dist, output_csv):
    print(f"Measuring SNR for {method} at {dist}mm...")

    cwd = getcwd()
    reps = [125, 250, 500, 1000, 1500, 2000]
    files = [f"current{num}reps_{dist}mm_{method}-lh.stc" for num in reps]
    
    stc_files = [path.join(cwd, p.data, p.processed_data, 
                          p.protocol, p.subj, p.session, 
                          p.current_dir, filename) for filename in files]

    stcs, valid_reps = [], []
    for i, f in enumerate(stc_files):
        if path.exists(f):
            stcs.append(mne.read_source_estimate(f))
            valid_reps.append(reps[i])
        else:
            print(f"Warning: {f} not found. Skipping.")

    if not stcs:
        print("Error: No valid STC files found.")
        return

    times = stcs[0].times
    idx_min = np.argmin(np.abs(times - p.tmin_snr))
    idx_max = np.argmin(np.abs(times - p.tmax_snr))

    snr_data_list = []

    for i, stc in enumerate(stcs):
        rep = valid_reps[i]
        for src_idx in range(stc.data.shape[0]):
            data = stc.data[src_idx][np.newaxis, :, np.newaxis]
            snr = snr_hammerer(p, data, times)
            snr_data_list.append({
                "rep": rep,
                "source": src_idx,
                "snr": float(snr[0])
            })

    df_all = pd.DataFrame(snr_data_list)
    output_path = path.join(cwd, p.data, p.processed_data, 
                          p.protocol, p.subj, p.session, 
                          p.current_dir, output_csv)
    df_all.to_csv(output_path, index=False)
    print(f"Saved SNR data to {output_path}")


def measure_snr_src_distspaces(p, method, output_csv):
    print("Measure SNR on source level for different source spaces: started")
    cwd = getcwd()
        
    reps  = [1000]
    dists = [2, 5, 10, 15, 20]
    methods = [method]
    
    files = [f"current{num}reps_{dist}mm_{method}-lh.stc" for num, dist, method in product(reps, dists, methods)]
    
    stc_files = [path.join(cwd, p.data, p.processed_data, 
                          p.protocol, p.subj, p.session, 
                          p.current_dir, filename) for filename in files]
    
    # Read STC files, only if they exist
    stcs = []
    valid_dists = []
    for i, f in enumerate(stc_files):
        if path.exists(f):
            stcs.append(mne.read_source_estimate(f))
            valid_dists.append(dists[i]) 
        else:
            print(f"Warning: {f} not found. Skipping.")
            
    if not stcs:
        print("Error: No valid STC files found.")
        return
    
    times = stcs[0].times
    idx_min = np.argmin(np.abs(times - p.tmin_snr))
    idx_max = np.argmin(np.abs(times - p.tmax_snr))
    
    # Compute SNR for the max activation source in each STC
    data_list = []
    for i, stc in enumerate(stcs):
        # Find the source with the max activation in this STC
        max_activation = np.max(np.abs(stc.data[:, idx_min:idx_max]), axis=1)
        max_source_idx = np.argmax(max_activation)
        
        if max_source_idx < len(stc.data):
            dist = valid_dists[i]  # Use valid distances
            
            # Ensure data has correct shape (1, timepoints, 1)
            data = stc.data[max_source_idx][np.newaxis, :, np.newaxis]
            snr = snr_hammerer(p, data, times)
            # Convert NumPy array to scalar
            snr_scalar = float(snr[0])  

            data_list.append({
                "dist": dist,
                "source": f"Source {max_source_idx} (Max Activation)",
                "snr": snr_scalar
            })
    
    df = pd.DataFrame(data_list)

    # Initialize the figure
    plt.figure(figsize=(10, 6))

    # Seaborn lineplot for SNR across different distances
    sns.lineplot(data=df, x="dist", y="snr", marker="o")
    
    # Customize the plot
    plt.xlabel("Distance of sources (mm)")
    plt.ylabel("SNR")
    plt.title("SNR vs. Source Distance for Max Activation Sources")
    #plt.legend(title="Source")
    plt.xlabel("Distance of sources (mm)", fontsize=18)
    plt.ylabel("SNR", fontsize=18)
    plt.title("SNR vs. Source Distance for Max Activation Sources", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)


    # Show the plot
    plt.show()

    # Save to CSV
    output_path = path.join(cwd, p.data, p.processed_data, 
                          p.protocol, p.subj, p.session, 
                          p.current_dir, output_csv)
    df.to_csv(output_path, index=False)
    
    print("Measure SNR on source level for different source spaces: completed")




#------------------------------------------------------------------------------------#    
def plot_snr_source_trend_multi_subject(p):
    print("Plotting source-level SNR across multiple subjects vs distance: started")
    cwd = getcwd()
    subjects = p.subjects
    methods = ["dSPM", "MNE", "eLORETA", "sLORETA"]
    distances = [5, 10, 15, 20]

    all_data = []

    for subj in subjects:
        subj_number = ''.join(filter(str.isdigit, subj))
        for method in methods:
            for dist in distances:
                filename = f"snr_{method}_{dist}mm_source.csv"
                filepath = path.join(cwd, p.data, p.processed_data,
                                     p.protocol, subj, p.session,
                                     p.current_dir, filename)

                if not path.exists(filepath):
                    print(f"Missing file for {subj}, {method}, {dist}mm. Skipping.")
                    continue

                df = pd.read_csv(filepath)
                df["Subject"] = subj
                df["Method"] = method
                df["Distance"] = dist
                all_data.append(df)

    if not all_data:
        print("No valid data found. Exiting.")
        return

    df_all = pd.concat(all_data)

    # Aggregate using the 95th percentile
    df_summary = (
        df_all
        .groupby(["Subject", "Method", "Distance"])["snr"]
        .quantile(0.95)
        .reset_index()
        .rename(columns={"snr": "snr_95th"})
    )

    # Plotting setup
    sns.set(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    method_order = ["dSPM", "MNE", "eLORETA", "sLORETA"]
    axes = axes.flatten()

    for i, method in enumerate(method_order):
        ax = axes[i]
        df_m = df_summary[df_summary["Method"] == method]

        # Individual subject lines in light grey
        sns.lineplot(
            data=df_m,
            x="Distance", y="snr_95th",
            hue="Subject", units="Subject",
            estimator=None, lw=1.2, alpha=0.4,
            color="lightgrey", legend=False, ax=ax
        )

        # Group average ± std in black
        sns.lineplot(
            data=df_m,
            x="Distance", y="snr_95th",
            estimator="mean", errorbar="sd",
            color="black", lw=2.5,
            label="Group Avg ± SD", ax=ax
        )

        ax.set_title(f"{method}", fontsize=16, weight="bold")
        ax.set_xlabel("Source Distance (mm)", fontsize=12)
        if i % 2 == 0:
            ax.set_ylabel("SNR (95th Percentile)", fontsize=12)
        else:
            ax.set_ylabel("")

        ax.grid(True, linestyle='--', alpha=0.5)

    fig.suptitle("SNR (95th Percentile) vs Distance — Source-Level Trends Across Subjects",
                 fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Shared legend
    custom_lines = [
        Line2D([0], [0], color='lightgrey', lw=1.2, alpha=0.4, label='Individual Subjects'),
        Line2D([0], [0], color='black', lw=2.5, label='Group Avg ± SD')
    ]
    fig.legend(
        handles=custom_lines,
        loc="upper left", bbox_to_anchor=(0.01, 0.99),
        fontsize=12
    )

    plt.show()
    print("Plotting source-level SNR across multiple subjects vs distance: completed")


def measure_snr_src_repnum(p, method, output_csv):
    print("Measure SNR on source level for different amounts of epochs: started")
    cwd = getcwd()
        
    reps  = [125,250,500,1000,1500,2000]
    dists = [20]
    methods = [method]
    
    files = [f"current{num}reps_{dist}mm_{method}-lh.stc" for num, dist, method in product(reps, dists, methods)]
    
    stc_files = [path.join(cwd, p.data, p.processed_data, 
                          p.protocol, p.subj, p.session, 
                          p.current_dir, filename) for filename in files]
    
    stcs = []
    valid_reps = []
    for i, f in enumerate(stc_files):
        if path.exists(f):
            stcs.append(mne.read_source_estimate(f))
            valid_reps.append(reps[i])
        else:
            print(f"Warning: {f} not found. Skipping.")

    if not stcs:
        print("Error: No valid STC files found.")
        return
    
    times = stcs[0].times
    idx_min = np.argmin(np.abs(times - p.tmin_snr))
    idx_max = np.argmin(np.abs(times - p.tmax_snr))
    
    snr_data_list = []  # For histogram and boxplot
    max_snr_list = []   # For line plot

    for i, stc in enumerate(stcs):
        rep = valid_reps[i]
        snrs = []

        for src_idx in range(stc.data.shape[0]):
            data = stc.data[src_idx][np.newaxis, :, np.newaxis]
            snr = snr_hammerer(p, data, times)
            snr_scalar = float(snr[0])
            snr_data_list.append({"rep": rep, "source": src_idx, "snr": snr_scalar})
            snrs.append(snr_scalar)

        max_snr_list.append({
            "rep": rep,
            "snr": max(snrs),
            "source": "Max Source"
        })

    df_all = pd.DataFrame(snr_data_list)
    df_max = pd.DataFrame(max_snr_list)

    # Save to CSV
    output_path = path.join(cwd, p.data, p.processed_data, 
                          p.protocol, p.subj, p.session, 
                          p.current_dir, output_csv)
    df_all.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print("Measure SNR on source level for different amounts of epochs: completed")


def plot_snr_ridges_across_methods(p):
    print("Generating SNR ridge plots for all methods and reps")
    methods = ["dSPM", "MNE", "eLORETA", "sLORETA"]
    dfs = []

    method_linestyles = {
        "dSPM": "dashdot",
        "MNE": "dashed",
        "eLORETA": "dotted",
        "sLORETA": "solid"
    }

    for method in methods:
        csv_name = f"snr_{method}_source.csv"
        full_path = path.join(getcwd(), p.data, p.processed_data,
                              p.protocol, p.subj, p.session,
                              p.current_dir, csv_name)
        if not path.exists(full_path):
            measure_snr_src_repnum(p, method, csv_name)

        df = pd.read_csv(full_path)
        df["method"] = method
        dfs.append(df)

    df_all = pd.concat(dfs)
    df_all["rep"] = df_all["rep"].astype(str)

    # Theme and palette
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    method_palette = dict(zip(methods, sns.color_palette("rocket", len(methods))))

    # Define the custom plotting function
    def ridgeplot(data, color, **kwargs):
        ax = plt.gca()
        peak_heights = {}
        kde_lines = {}

        for method in methods:
            subset = data[data["method"] == method]["snr"]
            if len(subset) >= 2:
                kde = sns.kdeplot(subset, bw_adjust=0.5, ax=ax, color=method_palette[method])
                line = kde.get_lines()[-1]
                x, y = line.get_data()
                peak_heights[method] = max(y)
                kde_lines[method] = (x, y)
                line.remove()
            else:
                kde_lines[method] = (np.array([0, 0]), np.array([0, 0]))
                peak_heights[method] = 0

        sorted_methods = sorted(peak_heights, key=peak_heights.get, reverse=True)
        ymax = 0

        for method in sorted_methods:
            x, y = kde_lines[method]
            ymax = max(ymax, max(y))
            ax.fill_between(x, y, alpha=0.1, color=method_palette[method])
             # Main line with unique style
            ax.plot(x, y, color=method_palette[method],
                    lw=2, alpha=0.8, linestyle=method_linestyles[method])
            
            #ax.plot(x, y, color="white", lw=0.1, alpha=0.5)


        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlim(df_all["snr"].min(), df_all["snr"].max())
        ax.set_xlabel("")

        # Add N = label
        rep = data["rep"].iloc[0]
        ax.text(df_all["snr"].min(), ymax * 1.05, f"N = {rep}",
                fontweight="bold", fontsize=9, color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    # Setup FacetGrid
    g = sns.FacetGrid(df_all, row="rep", aspect=15, height=0.6, sharex=True)
    g.map_dataframe(ridgeplot)

    g.fig.subplots_adjust(hspace=-0.55)
    g.set_titles("")
    g.despine(bottom=True, left=True)

    # Legend
    handles = [plt.Line2D([0], [0], color=method_palette[m], lw=4) for m in methods]
    g.fig.legend(handles, methods, title="Method", loc="upper right", bbox_to_anchor=(0.95, 0.99)).get_frame().set_visible(False)

    g.fig.suptitle("SNR Distributions Across Methods and Repetitions", fontsize=16, weight="bold")
    plt.show()


def plot_fixed_effects_with_significance(result, scale_rep=1000):
    """
    Plot fixed effects from LMM with 95% CI and significance asterisks.
    Optionally scale 'rep' effect (e.g. to per 1000 reps).
    """
    # Extract fixed effects, confidence intervals, and p-values
    fe_params = result.fe_params
    conf_int = result.conf_int()
    pvals = result.pvalues

    # Combine everything
    df_plot = pd.concat([
        fe_params.rename("estimate"),
        conf_int.rename(columns={0: "ci_low", 1: "ci_high"}),
        pvals.rename("pval")
    ], axis=1).drop("Intercept", errors='ignore').reset_index().rename(columns={"index": "term"})

    # Optionally rescale 'rep'
    if "rep" in df_plot["term"].values:
        df_plot.loc[df_plot["term"] == "rep", ["estimate", "ci_low", "ci_high"]] *= scale_rep
        df_plot.loc[df_plot["term"] == "rep", "term"] = f"rep (×{scale_rep})"

    # Label effect type
    def get_type(term):
        if "method" in term:
            return "Method"
        elif "distance" in term:
            return "Distance"
        elif "rep" in term:
            return "Repetition"
        return "Other"

    df_plot["type"] = df_plot["term"].apply(get_type)

    # Significance stars
    def get_significance(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    df_plot["signif"] = df_plot["pval"].apply(get_significance)

    # Sort for clarity
    df_plot = df_plot.sort_values("estimate", ascending=False).reset_index(drop=True)

    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_plot, x="estimate", y="term", hue="type",
        dodge=False, palette="rocket", errorbar=None
    )

    # Align error bars correctly
    y_positions = [p.get_y() + p.get_height() / 2 for p in ax.patches]

    for y, row in zip(y_positions, df_plot.itertuples()):
        err_low = row.estimate - row.ci_low
        err_high = row.ci_high - row.estimate
        ax.errorbar(
            x=row.estimate, y=y,
            xerr=[[err_low], [err_high]],
            fmt='none', ecolor='black', capsize=4, linewidth=1
        )

        # Add significance stars
        if row.signif:
            ax.text(
                row.estimate + 0.02, y,
                row.signif, color='black',
                va='center', ha='left', fontsize=12, weight='bold'
            )

    plt.axvline(0, linestyle="--", color="gray")
    plt.xlabel("Estimated Effect on SNR")
    plt.ylabel("Model Term")
    plt.title("Fixed Effects on SNR (with 95% CI and Significance)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
#-----------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------#    
def measure_snr_src_topoplot(p):
    print("Measure SNR for all sources and plot brain maps: started")
    
    cwd = getcwd()
    reps  = [125, 250, 500, 1000, 1500, 2000]
    dists = [5]
    methods = ['sLORETA']
    
    files = [f"current{num}reps_{dist}mm_{method}-lh.stc" for num, dist, method in product(reps, dists, methods)]
    
    stc_files = [path.join(cwd, p.data, p.processed_data, 
                          p.protocol, p.subj, p.session, 
                          p.current_dir, filename) for filename in files]
    
    stcs = []
    valid_reps = []
    for i, f in enumerate(stc_files):
        if path.exists(f):
            stcs.append(mne.read_source_estimate(f))
            valid_reps.append(reps[i])
        else:
            print(f"Warning: {f} not found. Skipping.")

    if not stcs:
        print("Error: No valid STC files found.")
        return
    
    times = stcs[0].times
   
    snr_stcs = [] 
    for i, stc in enumerate(stcs):
        rep = valid_reps[i]  
        data = stc.data
        snr = snr_hammerer(p, data, times)
        snr_stc = stc.copy()
        snr_stc.data = snr[:, np.newaxis]
        snr_stcs.append((snr_stc, rep))
    
    brain_images = []  # Store screenshots and their corresponding color limits
    
    for i, (snr_stc, rep) in enumerate(snr_stcs):
        print(f"Plotting SNR for {rep} reps...")

        # Unique color limits for each brain plot
        clim = dict(kind='value', lims=[
            np.min(snr_stc.data),
            np.median(snr_stc.data),
            np.max(snr_stc.data)
        ])
        
        brain = snr_stc.plot(
            subject=p.fs_subject_dir,
            subjects_dir=p.subj_brains_dir,
            surface='inflated',
            hemi='both',
            colormap='rocket',
            clim=clim,
            background='white',
            time_viewer=False,
            size=(600, 600),
            colorbar=False
        )

        brain.show_view(azimuth=180, elevation=0)
        screenshot = brain.screenshot()
        brain_images.append((screenshot, rep, clim))
        brain.close()


    
    # Plot all screenshots in a single matplotlib figure with individual colorbars
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    for ax, (img, rep, clim) in zip(axes.flatten(), brain_images):
        im = ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{rep} Epochs", fontsize=14)

        # Add vertical colorbar on the right
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        norm = plt.Normalize(clim['lims'][0], clim['lims'][2])
        sm = plt.cm.ScalarMappable(cmap='rocket', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cax)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

    print("Measure SNR for all sources and plot brain maps: completed")



#-----------------------------------------------------------------------------------#
def measure_snr_src_multi_subject(p, method, filter_s1_only=False):
    print("Measure SNR on source level for multiple subjects: started")
    cwd = getcwd()

    reps = [125, 250, 500, 1000, 1500, 2000]
    dists = [10]
    subjects = p.subjects

    all_snr_data = []

    # Define S1 ROI patterns
    s1_patterns = [
        'L_3b_ROI-lh', 'L_1_ROI-lh', 'L_2_ROI-lh',
        'R_3b_ROI-rh', 'R_1_ROI-rh', 'R_2_ROI-rh',
        'L_4_ROI-lh', 'R_4_ROI-rh',
        'L_6d_ROI-lh', 'R_6d_ROI-rh',
        'L_6v_ROI-lh', 'R_6v_ROI-rh',
        'L_6a_ROI-lh', 'R_6a_ROI-rh',
        'L_6mp_ROI-lh', 'R_6mp_ROI-rh',
        'L_5m_ROI-lh', 'R_5m_ROI-rh',
        'L_5L_ROI-lh', 'R_5L_ROI-rh'
    ]

    # Get S1 source mask if filtering is requested
    tmp_filename = f"current{reps[0]}reps_{dists[0]}mm_{method}-lh.stc"
    tmp_path = path.join(cwd, p.data, p.processed_data, p.protocol,
                     subjects[0], p.session, p.current_dir, tmp_filename)
    
    s1_mask = None
    if filter_s1_only:
        if path.exists(tmp_path):
            reference_stc = mne.read_source_estimate(tmp_path)
            s1_mask = get_s1_source_mask(p, s1_patterns, reference_stc)
        else:
            print(f"Reference STC file not found: {tmp_path}")
            s1_mask = None     
        if s1_mask is None or not np.any(s1_mask):
            print("Warning: No S1 sources found. Falling back to all sources.")
            filter_s1_only = False
        else:
            print(f"Filtering to {np.sum(s1_mask)} S1 sources out of {len(s1_mask)} total")

    for subj in subjects:
        subj_snr_data = []
        stcs = []
        valid_reps = []

        for rep in reps:
            filename = f"current{rep}reps_{dists[0]}mm_{method}-lh.stc"
            stc_path = path.join(cwd, p.data, p.processed_data, p.protocol,
                                 subj, p.session, p.current_dir, filename)
            if path.exists(stc_path):
                stcs.append(mne.read_source_estimate(stc_path))
                valid_reps.append(rep)
            else:
                print(f"Missing STC for {subj}, {rep} reps. Skipping.")

        if not stcs:
            print(f"No valid STC files for subject {subj}. Skipping.")
            continue

        times = stcs[0].times
        idx_min = np.argmin(np.abs(times - p.tmin_snr))
        idx_max = np.argmin(np.abs(times - p.tmax_snr))

        # Find the source with maximum activation, optionally restricted to S1
        if filter_s1_only and s1_mask is not None:
            # Apply S1 mask to find max activation only within S1
            stc_data_masked = stcs[-1].data[s1_mask, :]
            max_activation_masked = np.max(np.abs(stc_data_masked[:, idx_min:idx_max]), axis=1)
            max_source_idx_in_mask = np.argmax(max_activation_masked)
            # Convert back to original source space index
            max_source_idx = np.where(s1_mask)[0][max_source_idx_in_mask]
        else:
            # Original logic for all sources
            max_activation = np.max(np.abs(stcs[-1].data[:, idx_min:idx_max]), axis=1)
            max_source_idx = np.argmax(max_activation)

        for i, stc in enumerate(stcs):
            data = stc.data[max_source_idx][np.newaxis, :, np.newaxis]
            snr = snr_hammerer(p, data, times)
            snr_scalar = float(snr[0])
            subj_snr_data.append({
                "Repetitions": valid_reps[i],
                "SNR": snr_scalar,
                "Subject": subj
            })

        all_snr_data.extend(subj_snr_data)

    if not all_snr_data:
        print("No SNR data collected. Exiting.")
        return

    df = pd.DataFrame(all_snr_data)

    # Compute average and std
    avg_df = df.groupby("Repetitions", as_index=False)["SNR"].mean()
    std_df = df.groupby("Repetitions", as_index=False)["SNR"].std().rename(columns={"SNR": "STD"})
    avg_df["Subject"] = "Average"
    avg_df = avg_df.merge(std_df, on="Repetitions")

    # Combine with individual data
    df_combined = pd.concat([df, avg_df])

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    for subj in df["Subject"].unique():
        if subj == "Average":
            continue
        subj_df = df[df["Subject"] == subj]
        plt.plot(
            subj_df["Repetitions"], subj_df["SNR"],
            color="gray", alpha=0.3, linewidth=1.5, zorder=1
        )

    plt.plot(
        avg_df["Repetitions"], avg_df["SNR"],
        color="black", linewidth=2.5, marker='D', label="Average ± Std", zorder=2
    )
    plt.fill_between(
        avg_df["Repetitions"],
        avg_df["SNR"] - avg_df["STD"],
        avg_df["SNR"] + avg_df["STD"],
        color="black", alpha=0.2
    )

    # Update title to reflect S1 filtering
    title = "SNR vs. Repetitions Across Subjects"
    if filter_s1_only:
        title += " (S1 ROIs Only)"
    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel("Number of trials", fontsize=16)
    plt.ylabel("SNR", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)

    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, alpha=0.5, label='Subjects'),
        Line2D([0], [0], color='black', lw=2.5, marker='D', label='Group Average ± Std')
    ]
    plt.legend(handles=legend_elements, fontsize=14)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Pairwise Wilcoxon Tests with FDR correction (applied to filtered data)
    from scipy.stats import wilcoxon
    from matplotlib.colors import TwoSlopeNorm
    import itertools
    
    pivot_df = df.pivot(index="Subject", columns="Repetitions", values="SNR")
    pivot_df = pivot_df.dropna()  # Drop subjects with missing values
    reps_available = sorted(pivot_df.columns.tolist())
    
    if len(pivot_df) == 0:
        print("Not enough subjects with complete data. Skipping statistical tests.")
    else:
        print(f"Applying Wilcoxon tests to {len(pivot_df)} subjects")
        if filter_s1_only:
            print("Statistical tests applied to S1 ROIs only")
        
        # Collect Wilcoxon results
        p_values = []
        comparisons = []
        for rep1, rep2 in itertools.combinations(reps_available, 2):
            try:
                stat, pval = wilcoxon(pivot_df[rep1], pivot_df[rep2])
                p_values.append(pval)
                comparisons.append((rep1, rep2))
            except ValueError:
                p_values.append(np.nan)
                comparisons.append((rep1, rep2))
    
        # Multiple comparison correction (FDR)
        reject, pvals_corr, _, _ = multipletests(p_values, method="fdr_bh")
    
        # Build corrected p-value matrix
        pval_matrix_corr = pd.DataFrame(
            np.ones((len(reps_available), len(reps_available))),
            index=reps_available, columns=reps_available
        )
        for (rep1, rep2), p_corr in zip(comparisons, pvals_corr):
            pval_matrix_corr.loc[rep1, rep2] = p_corr
            pval_matrix_corr.loc[rep2, rep1] = p_corr
    
        np.fill_diagonal(pval_matrix_corr.values, 1.0)
    
        # Print table of results
        print(f"{'Comparison':<15}{'Corrected p':<15}{'Significant':<12}")
        print("-" * 45)
        for (rep1, rep2), p_corr, rej in zip(comparisons, pvals_corr, reject):
            if np.isnan(p_corr):
                continue
            print(f"{rep1} vs {rep2:<7}{p_corr:<15.4f}{str(rej):<12}")
    
        # --- Plot heatmap of corrected p-values ---
        mask = np.triu(np.ones_like(pval_matrix_corr, dtype=bool))
        significance_level = 0.05  # corrected significance level
        
        # Diverging colormap centered on significance threshold (0.05)
        cmap = plt.cm.RdBu_r
        norm = TwoSlopeNorm(vmin=0, vcenter=significance_level, vmax=1)
        
        plt.figure(figsize=(9, 7))
        ax = sns.heatmap(
            pval_matrix_corr,
            mask=mask,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            norm=norm,
            cbar_kws={"label": "Corrected p-value"},
            linewidths=0.5,
            linecolor='gray',
            annot_kws={"size": 18},
            square=True
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)      # tick labels
        cbar.ax.yaxis.label.set_size(22)       # colorbar label
                
        # Update title to reflect S1 filtering
        heatmap_title = f"Wilcoxon p-values (corrected) — {method}"
        if filter_s1_only:
            heatmap_title += " (S1 ROIs)"
        plt.title(heatmap_title, fontsize=20, weight="bold")
        plt.xlabel("Number of trials", fontsize=20)
        plt.ylabel("Number of trials", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.show()

    print("Measure SNR on source level for multiple subjects: completed")


def get_s1_source_mask(p, s1_patterns, reference_stc):
    """
    Create a boolean mask for S1 sources using MNE's label functionality.
    
    Parameters:
    -----------
    p : parameter object
        Contains paths and subject information
    s1_patterns : list
        List of S1 ROI pattern strings
    reference_stc : SourceEstimate
        Reference STC to get the exact source space structure
    
    Returns:
    --------
    numpy.ndarray : Boolean mask for S1 sources, or None if failed
    """
   
    from os.path import join
    
    try:
        cwd = getcwd()
        subjdirpath = join(cwd, p.subj_brains_dir)
        
        # Read labels from HCPMMP1 parcellation
        labels = mne.read_labels_from_annot(
            subject=p.fs_subject_dir,
            parc='HCPMMP1',
            hemi='both',
            subjects_dir=subjdirpath
        )
        
        # Filter to S1 labels only
        s1_labels = [label for label in labels if label.name in s1_patterns]
        
        if not s1_labels:
            print(f"Warning: No S1 labels found matching patterns: {s1_patterns}")
            available_labels = [label.name for label in labels if any(roi in label.name.lower() 
                               for roi in ['3b', '1_', '2_', '4_', '5', '6'])]
            print(f"Available similar labels: {available_labels[:10]}...")
            return None
        
        print(f"Found {len(s1_labels)} S1 labels: {[label.name for label in s1_labels]}")
        
        # Get source space structure from reference STC
        n_sources = reference_stc.data.shape[0]
        vertices_lh = reference_stc.vertices[0]
        vertices_rh = reference_stc.vertices[1]
        
        # Create boolean mask
        mask = np.zeros(n_sources, dtype=bool)
        
        # Process each S1 label
        current_idx = 0
        
        # Left hemisphere sources
        for vertex in vertices_lh:
            # Check if this vertex belongs to any S1 label
            for label in s1_labels:
                if label.hemi == 'lh' and vertex in label.vertices:
                    mask[current_idx] = True
                    break
            current_idx += 1
        
        # Right hemisphere sources  
        for vertex in vertices_rh:
            # Check if this vertex belongs to any S1 label
            for label in s1_labels:
                if label.hemi == 'rh' and vertex in label.vertices:
                    mask[current_idx] = True
                    break
            current_idx += 1
        
        print(f"Created mask for {n_sources} sources: {np.sum(mask)} S1 sources found")
        return mask
        
    except Exception as e:
        print(f"Error creating S1 source mask: {e}")
        return None
#----------------------------------------------------------------------------------------#
def plot_mse_wilcoxon_heatmaps(df, mse_column="MSE_All"):
    reps = sorted(df["Repetitions"].unique())

    for method in df["Method"].unique():
        for dist in sorted(df["Distance"].unique()):
            sub_df = df[(df["Method"] == method) & (df["Distance"] == dist)]
            pivot = sub_df.pivot(index="Subject", columns="Repetitions", values=mse_column)
            pivot = pivot.dropna()

            if pivot.shape[0] < 3:
                print(f"Skipping {method} @ {dist}mm (not enough subjects)")
                continue

            # Initialize matrix
            pval_matrix = pd.DataFrame(index=reps, columns=reps, dtype=float)

            for i, rep1 in enumerate(reps):
                for j, rep2 in enumerate(reps):
                    if rep1 == rep2:
                        pval_matrix.loc[rep1, rep2] = 1.0
                    else:
                        try:
                            data1 = pivot[rep1]
                            data2 = pivot[rep2]
                            paired = data1.dropna().index.intersection(data2.dropna().index)
                            stat, p = wilcoxon(data1.loc[paired], data2.loc[paired])
                            pval_matrix.loc[rep1, rep2] = p
                        except ValueError:
                            pval_matrix.loc[rep1, rep2] = np.nan

            # Plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                pval_matrix.astype(float),
                annot=True,
                fmt=".3f",
                cmap="coolwarm",
                vmin=0, vmax=1,
                linewidths=0.5,
                linecolor='gray',
                cbar_kws={"label": "p-value"}
            )
            plt.title(f"Wilcoxon p-values\n{method} @ {dist}mm", fontsize=16, weight='bold')
            plt.xlabel("Repetitions")
            plt.ylabel("Repetitions")
            plt.tight_layout()
            plt.show()


def normalize_stc_data(data, method):
    if method in ['dSPM', 'sLORETA']:
        # Normalize each source estimate to have unit Frobenius norm
        norm = np.linalg.norm(data)
        if norm != 0:
            return data / norm
        else:
            return data  # Avoid division by zero
    else:
        return data

def measure_mse_between_splits(p):
    print("Measure MSE between split source estimates: started")
    cwd = getcwd()

    reps = [65, 125, 250, 500, 750, 1000]
    dists = [5, 10, 15, 20]
    methods = ['eLORETA', 'sLORETA', 'MNE', 'dSPM']
    splits = ['split1', 'split2']
    subjects = p.subjects

    all_mse_data = []

    for method in methods:
        for dist in dists:
            for subj in subjects:
                for rep in reps:
                    stcs = {}

                    for split in splits:
                        filename = f"current{rep}reps_{dist}mm_{method}_{split}-lh.stc"
                        filepath = path.join(cwd, p.data, p.processed_data,
                                             p.protocol, subj, p.session,
                                             p.current_dir, filename)
                        if path.exists(filepath):
                            stcs[split] = mne.read_source_estimate(filepath)
                        else:
                            print(f"Missing file: {filename}")
                            break

                    if len(stcs) < 2:
                        continue  # Skip if one of the splits is missing

                    # Compute MSE between the two splits
                    data1 = stcs['split1'].data
                    data2 = stcs['split2'].data

                    # Normalize
                    data1 = normalize_stc_data(stcs['split1'].data, method)
                    data2 = normalize_stc_data(stcs['split2'].data, method)

                    mse_all = np.mean((data1 - data2) ** 2)

                    times = stcs['split1'].times
                    idx_min = np.argmin(np.abs(times - p.tmin_snr))
                    idx_max = np.argmin(np.abs(times - p.tmax_snr))

                    avg_data = (np.abs(data1) + np.abs(data2)) / 2
                    max_activation = np.max(avg_data[:, idx_min:idx_max], axis=1)
                    max_idx = np.argmax(max_activation)

                    mse_peak = np.mean((data1[max_idx, :] - data2[max_idx, :]) ** 2)

                    all_mse_data.append({
                        "Subject": subj,
                        "Repetitions": rep,
                        "Distance": dist,
                        "Method": method,
                        "MSE_All": mse_all,
                        "MSE_Peak": mse_peak
                    })

    if not all_mse_data:
        print("No MSE data collected.")
        return

    df = pd.DataFrame(all_mse_data)
    sns.set(style="whitegrid")

    # ---- Plotting with Subplots ----
    for metric in ['MSE_All', 'MSE_Peak']:
        for method in methods:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
            fig.suptitle(f"{metric} vs Repetitions — Method: {method}", fontsize=18, weight='bold')

            for i, dist in enumerate(dists):
                ax = axes[i]
                sub_df = df[(df.Method == method) & (df.Distance == dist)]

                for subj in sub_df.Subject.unique():
                    subj_df = sub_df[sub_df.Subject == subj]
                    ax.plot(
                        subj_df["Repetitions"], subj_df[metric],
                        color='gray', alpha=0.3, linewidth=1.2
                    )

                # Group average and std
                mean_df = sub_df.groupby("Repetitions")[metric].mean().reset_index()
                std_df = sub_df.groupby("Repetitions")[metric].std().reset_index()

                ax.plot(
                    mean_df["Repetitions"], mean_df[metric],
                    color='black', linewidth=2.5, marker='o',
                    label="Average ± Std"
                )
                ax.fill_between(
                    mean_df["Repetitions"],
                    mean_df[metric] - std_df[metric],
                    mean_df[metric] + std_df[metric],
                    alpha=0.2, color='black'
                )

                ax.set_title(f"{dist} mm", fontsize=24)
                ax.set_xlabel("Number of trials", fontsize=20)
                ax.tick_params(axis="both", labelsize=20)  # Change from 10 to whatever size you want
                if i == 0:
                    ax.set_ylabel("MSE (μV²)", fontsize=20)
                ax.grid(True, linestyle='--', alpha=0.4)

            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', fontsize=18)
            plt.tight_layout(rect=[0, 0, 0.97, 0.93])  # Leave space for title and legend
            plt.show()

    # perform statistical test
    plot_mse_wilcoxon_heatmaps(df, mse_column="MSE_All")
    
    print("Measure MSE between split source estimates: completed")
#-----------------------------------------------------------------------------------------#
def plot_snr_hexbin_with_marginals(p, input_csv_el, input_csv_src, output_path=None, rep_filter=None):
    """
    Loads two CSV files and creates a hexbin plot with marginal distributions
    comparing all `snr` values from source-level data and all `snr_mean` values 
    from electrode-level data for a given repetition count.
    """
    if rep_filter is None:
        raise ValueError("You must provide a single rep_filter value (e.g., 500)")

    # Construct file paths
    cwd = getcwd()
    file_snr_src = path.join(cwd, p.data, p.processed_data, 
                             p.protocol, p.subj, p.session, 
                             p.current_dir, input_csv_src)
    file_snr_el = path.join(cwd, p.data, p.processed_data, 
                            p.protocol, p.subj, p.session, 
                            p.current_dir, input_csv_el)

    # Load data
    df_src = pd.read_csv(file_snr_src)
    df_el = pd.read_csv(file_snr_el)

    # Filter both datasets for the selected repetition count
    df_src_filtered = df_src[df_src['rep'] == rep_filter].copy()
    df_el_filtered = df_el[df_el['n_select'] == rep_filter].copy()

    # Expand electrode-level rows so each SNR mean becomes its own row
    df_el_expanded = df_el_filtered[['channel', 'snr_mean']].copy()
    df_el_expanded.rename(columns={'snr_mean': 'snr_el'}, inplace=True)

    df_src_expanded = df_src_filtered[['source', 'snr']].copy()
    df_src_expanded.rename(columns={'snr': 'snr_src'}, inplace=True)

    # Create all combinations between electrode SNRs and source SNRs
    merged_df = df_el_expanded.assign(key=1).merge(
        df_src_expanded.assign(key=1), on='key'
    ).drop(columns='key')

    # Plot
    sns.set(style="white", font_scale=1.2)
    g = sns.jointplot(
        data=merged_df,
        x='snr_src',
        y='snr_el',
        kind='hex',
        gridsize=40,
        cmap="cividis",
        marginal_kws=dict(bins=30, fill=True)
    )

    # Set axis labels
    g.set_axis_labels("Source SNR", "Electrode SNR", fontsize=14)

    # Plot diagonal
    xlim = g.ax_joint.get_xlim()
    ylim = g.ax_joint.get_ylim()
    min_val = min(xlim[0], ylim[0])
    max_val = max(xlim[1], ylim[1])
    g.ax_joint.plot([min_val, max_val], [min_val, max_val], 
                    linestyle='--', color='lightgray', linewidth=1.5)
    g.ax_joint.set_xlim(xlim)
    g.ax_joint.set_ylim(ylim)

    # Force redraw hexbin with mincnt=1, empty bins (N=0) shown as black
    for artist in g.ax_joint.collections:
        artist.remove()
    
    hb = g.ax_joint.hexbin(
        merged_df['snr_src'], merged_df['snr_el'],
        gridsize=40, cmap="cividis", mincnt=1
    )

    # Set background color to black so empty bins show as black
    g.ax_joint.set_facecolor("black")

    # Add colorbar from the new hexbin
    #cb = plt.colorbar(hb, ax=g.ax_joint)
    #cb.set_label("Counts (N ≥ 1)")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()



def plot_avg_snr_hexbin_with_marginals(p, input_csv_el_template, input_csv_src_template, output_path=None, rep_filter=None):
    """
    Aggregates electrode-level and source-level SNR data across all subjects
    and creates a hexbin plot with marginal distributions.
    
    Parameters:
    - p: configuration or parameter object
    - input_csv_el_template: string with '{}' placeholder for subject number, e.g., "snr_{}_entropy_el.csv"
    - input_csv_src: filename for source-level CSV (assumed the same for all subjects)
    - output_path: optional path to save the figure
    - rep_filter: repetition count to filter the data on (e.g., 500)
    """

    if rep_filter is None:
        raise ValueError("You must provide a single rep_filter value (e.g., 500)")

    from glob import glob

    cwd = getcwd()
    merged_df_list = []

    for subj in p.subjects:
        subj_number = ''.join(filter(str.isdigit, subj))  # extract numeric part of subject
        input_csv_el  = input_csv_el_template.format(subj_number)
        
        input_csv_src = input_csv_src_template.format(subj_number)

        file_snr_src = path.join(cwd, p.data, p.processed_data,
                                 p.protocol, subj, p.session,
                                 p.current_dir, input_csv_src)
        
        file_snr_el = path.join(cwd, p.data, p.processed_data,
                                p.protocol, subj, p.session,
                                p.current_dir, input_csv_el)

        if not path.exists(file_snr_src) or not path.exists(file_snr_el):
            print(f"Skipping {subj}: missing file(s)")
            continue

        # Load data
        df_src = pd.read_csv(file_snr_src)
        df_el = pd.read_csv(file_snr_el)

        # Filter for repetition
        df_src_filtered = df_src[df_src['rep'] == rep_filter].copy()
        df_el_filtered = df_el[df_el['n_select'] == rep_filter].copy()

        # Prepare
        df_el_expanded = df_el_filtered[['channel', 'snr_mean']].copy()
        df_el_expanded.rename(columns={'snr_mean': 'snr_el'}, inplace=True)

        df_src_expanded = df_src_filtered[['source', 'snr']].copy()
        df_src_expanded.rename(columns={'snr': 'snr_src'}, inplace=True)

        # Cartesian product
        merged = df_el_expanded.assign(key=1).merge(
            df_src_expanded.assign(key=1), on='key'
        ).drop(columns='key')

        merged_df_list.append(merged)

    if not merged_df_list:
        raise ValueError("No valid data found for any subjects.")

    # Concatenate all merged data
    all_merged_df = pd.concat(merged_df_list, ignore_index=True)

    # Plot
    sns.set(style="white", font_scale=1.2)
    g = sns.jointplot(
        data=all_merged_df,
        x='snr_src',
        y='snr_el',
        kind='hex',
        gridsize=40,
        cmap="cividis",
        marginal_kws=dict(bins=40, fill=True)
    )

    g.set_axis_labels("Source SNR", "Electrode SNR", fontsize=14)

    # Diagonal
    xlim = g.ax_joint.get_xlim()
    ylim = g.ax_joint.get_ylim()
    min_val = min(xlim[0], ylim[0])
    max_val = max(xlim[1], ylim[1])
    g.ax_joint.plot([min_val, max_val], [min_val, max_val],
                    linestyle='--', color='lightgray', linewidth=1.5)
    g.ax_joint.set_xlim(xlim)
    g.ax_joint.set_ylim(ylim)

    # Redraw hexbin with mincnt=1
    for artist in g.ax_joint.collections:
        artist.remove()

    hb = g.ax_joint.hexbin(
        all_merged_df['snr_src'], all_merged_df['snr_el'],
        gridsize=80, cmap="cividis", mincnt=1
    )
    g.ax_joint.set_facecolor("black")

    # Optional colorbar
    # cb = plt.colorbar(hb, ax=g.ax_joint)
    # cb.set_label("Counts (N ≥ 1)")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_snr_violin_per_subject_and_avg(p, input_csv_el_template, input_csv_src_template, rep_filter=None, output_path=None):
    """
    Aggregates electrode-level and source-level SNR data across all subjects,
    and creates violin plots comparing their distributions for each subject 
    and an overall average violin plot.
    
    Parameters:
    - p: configuration object with subject list and path info
    - input_csv_el_template: template like "snr_{}_entropy_el.csv"
    - input_csv_src_template: template like "snr_{}_entropy_src.csv"
    - rep_filter: repetition value to filter on (e.g., 500)
    - output_path: optional path to save the figure
    """

    if rep_filter is None:
        raise ValueError("You must provide a rep_filter (e.g., 500)")

    

    cwd = getcwd()
    all_data = []

    for subj in p.subjects:
        subj_number = ''.join(filter(str.isdigit, subj))  # Extract numeric part
        input_csv_el = input_csv_el_template.format(subj_number)
        input_csv_src = input_csv_src_template.format(subj_number)

        file_el = path.join(cwd, p.data, p.processed_data, p.protocol, subj, p.session, p.current_dir, input_csv_el)
        file_src = path.join(cwd, p.data, p.processed_data, p.protocol, subj, p.session, p.current_dir, input_csv_src)

        if not path.exists(file_el) or not path.exists(file_src):
            print(f"Skipping {subj}: missing one or both files")
            continue

        df_el = pd.read_csv(file_el)
        df_src = pd.read_csv(file_src)

        df_el_filtered = df_el[df_el['n_select'] == rep_filter].copy()
        df_src_filtered = df_src[df_src['rep'] == rep_filter].copy()

        if df_el_filtered.empty or df_src_filtered.empty:
            print(f"Skipping {subj}: no data for rep={rep_filter}")
            continue

        el_data = df_el_filtered[['snr_mean']].copy()
        el_data['type'] = 'Channel'
        el_data.rename(columns={'snr_mean': 'snr'}, inplace=True)
        el_data['subject'] = subj

        src_data = df_src_filtered[['snr']].copy()
        src_data['type'] = 'Source'
        src_data.rename(columns={'snr': 'snr'}, inplace=True)
        src_data['subject'] = subj

        all_data.append(pd.concat([el_data, src_data], axis=0))

    if not all_data:
        raise ValueError("No valid data found across subjects.")

    df_all = pd.concat(all_data, ignore_index=True)

    # --- Plot per subject ---
    sns.set(style="whitegrid", font_scale=1.1)
    g = sns.catplot(
        data=df_all,
        x='type',
        y='snr',
        col='subject',
        kind='violin',
        inner='quartile',
        cut=0,
        height=4,
        aspect=0.8,
        palette='rocket'
    )
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("", "SNR Value")
    g.fig.suptitle(f"SNR Distribution per Subject (rep={rep_filter})", fontsize=16)
    g.fig.subplots_adjust(top=0.85)

    # --- Plot aggregate ---
    plt.figure(figsize=(6, 5))
    sns.violinplot(
        data=df_all,
        x='type',
        y='snr',
        inner='quartile',
        cut=0,
        palette='rocket'
    )
    plt.title(f"Average SNR Distribution Across Subjects (rep={rep_filter})")
    plt.ylabel("SNR Value")
    plt.xlabel("")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_snr_violin_all_methods_and_avg(p, rep_filter=2000, output_path=None):
    """
    Generates violin plots comparing SNR distributions from electrode space and 
    multiple source methods (dSPM, MNE, eLORETA, sLORETA) for each subject, as 
    well as an aggregate plot across all subjects.

    Parameters:
    - p: configuration object with subject list and path info
    - rep_filter: repetition value to filter on (default=2000)
    - output_path: optional path to save the average figure
    """

    methods = ["dSPM", "MNE", "eLORETA", "sLORETA"]
    cwd = getcwd()
    all_data = []

    for subj in p.subjects:
        subj_number = ''.join(filter(str.isdigit, subj))

        # Electrode space file
        input_csv_el = f"snr_{subj_number}_el.csv"
        file_el = path.join(cwd, p.data, p.processed_data, p.protocol, subj, p.session, p.current_dir, input_csv_el)

        if not path.exists(file_el):
            print(f"Missing electrode file for {subj}")
            continue

        df_el = pd.read_csv(file_el)
        df_el_filtered = df_el[df_el['n_select'] == rep_filter].copy()

        if df_el_filtered.empty:
            print(f"No electrode data for {subj} at rep={rep_filter}")
            continue

        el_data = df_el_filtered[['snr_mean']].copy()
        el_data['method'] = 'Electrode'
        el_data.rename(columns={'snr_mean': 'snr'}, inplace=True)
        el_data['subject'] = subj

        subj_data = [el_data]

        for method in methods:
            input_csv_src = f"snr_{method}_source.csv"
            file_src = path.join(cwd, p.data, p.processed_data, p.protocol, subj, p.session, p.current_dir, input_csv_src)

            if not path.exists(file_src):
                print(f"Missing source file ({method}) for {subj}")
                continue

            df_src = pd.read_csv(file_src)
            df_src_filtered = df_src[df_src['rep'] == rep_filter].copy()

            if df_src_filtered.empty:
                print(f"No {method} data for {subj} at rep={rep_filter}")
                continue

            src_data = df_src_filtered[['snr']].copy()
            src_data['method'] = method
            src_data['subject'] = subj

            subj_data.append(src_data)

        if subj_data:
            all_data.append(pd.concat(subj_data, axis=0))

    if not all_data:
        raise ValueError("No valid data found across subjects.")

    df_all = pd.concat(all_data, ignore_index=True)

    # --- Plot per subject ---
    sns.set(style="whitegrid", font_scale=1.1)
    g = sns.catplot(
        data=df_all,
        x='method',
        y='snr',
        col='subject',
        kind='violin',
        inner='quartile',
        cut=0,
        height=4,
        aspect=0.8,
        palette='rocket'
    )
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("", "SNR Value")
    g.fig.suptitle(f"SNR Distribution per Subject (rep={rep_filter})", fontsize=16)
    g.fig.subplots_adjust(top=0.85)

    # --- Plot aggregate ---
    plt.figure(figsize=(6, 5))
    sns.violinplot(
        data=df_all,
        x='method',
        y='snr',
        inner='quartile',
        cut=0,
        palette='rocket'
    )
    plt.title(f"Average SNR Distribution Across Subjects (rep={rep_filter})")
    plt.ylabel("SNR Value")
    plt.xlabel("")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()
#-----------------------------------------------------------------------------------------#
def parse_method_distance(filename):
    """Extract method and distance from filenames like 'snr_eLORETA_10mm_source.csv'"""
    match = re.search(r"snr_([a-zA-Z]+)_(\d+)mm", filename)
    if not match:
        raise ValueError(f"Could not parse method/distance from: {filename}")
    return match.group(1), int(match.group(2))



def create_vertex_to_roi_dict(p, dist):
    """Return a mapping from (hemi, vertex) → ROI name for all sources."""
    cwd = getcwd()
    subjdirpath = path.join(cwd, p.subj_brains_dir)
    os.environ['SUBJECTS_DIR'] = subjdirpath

    # Load forward solution and source space
    fwdloadpath = os.path.join(
        cwd, p.data, p.processed_data, p.protocol,
        p.subj, p.session, p.leadfield_dir, f'subject_{dist}-fwd.fif'
    )
    src = mne.read_forward_solution(fwdloadpath)['src']

    # Load labels (HCPMMP1 atlas)
    labels = mne.read_labels_from_annot(
        subject=p.fs_subject_dir,
        parc='HCPMMP1',
        hemi='both',
        subjects_dir=subjdirpath
    )

    vertex_to_roi = {}
    for hemi_idx, hemi_str in enumerate(['lh', 'rh']):
        src_verts = src[hemi_idx]['vertno']
        label_lookup = {v: label.name for label in labels if label.hemi == hemi_str for v in label.vertices}
        for v in src_verts:
            roi = label_lookup.get(v, 'unknown')
            vertex_to_roi[(hemi_str, v)] = roi

    return vertex_to_roi, src


def load_all_snr_data(p, methods, dists, file_template="snr_{method}_{dist}mm_source.csv"):
    """Load all SNR data for all methods, distances, and subjects, with ROI labels."""
    all_dfs = []

    cwd = getcwd()

    for method in methods:
        for dist in dists:
            filename = file_template.format(method=method, dist=dist)

            # Build vertex→ROI mapping once per dist
            vertex_to_roi, src = create_vertex_to_roi_dict(p, dist)

            # Build vertex→index lookup for stc indexing
            vertno_lh = src[0]['vertno']
            vertno_rh = src[1]['vertno']
            vertex_to_index = {}
            for i, v in enumerate(vertno_lh):
                vertex_to_index[i] = ('lh', v)
            for i, v in enumerate(vertno_rh, start=len(vertno_lh)):
                vertex_to_index[i] = ('rh', v)

            for subj in p.subjects:
                subj_number = ''.join(filter(str.isdigit, subj))

                file_path = path.join(cwd, p.data, p.processed_data,
                                         p.protocol, subj, p.session,
                                         p.current_dir, filename)

                if not path.exists(file_path):
                    print(f"Missing: {file_path}")
                    continue

                try:
                    df = pd.read_csv(file_path)

                    # --- Add ROI column ---
                    if 'source' in df.columns:  # assuming your CSV has source indices
                        rois = []
                        for src_idx in df['source']:
                            hemi_v = vertex_to_index.get(src_idx)
                            if hemi_v is None:
                                rois.append('unknown')
                            else:
                                rois.append(vertex_to_roi.get(hemi_v, 'unknown'))
                        df['roi'] = rois
                    else:
                        print(f"Warning: no 'source' column in {file_path}")
                        df['roi'] = 'unknown'

                    # Add metadata
                    df['method'] = method
                    df['distance'] = dist
                    df['subject'] = subj_number

                    all_dfs.append(df)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

    if not all_dfs:
        raise ValueError("No SNR data found. Check filenames and paths.")

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure correct data types
    full_df['method'] = full_df['method'].astype('category')
    full_df['distance'] = full_df['distance'].astype('category')
    full_df['subject'] = full_df['subject'].astype('category')
    full_df['rep'] = pd.to_numeric(full_df['rep'], errors='coerce')
    full_df['roi'] = full_df['roi'].astype('category')

    return full_df



def load_all_snr_with_electrode(p, methods, dists,
                                 source_file_template="snr_{method}_{dist}mm_source.csv",
                                 electrode_file_template="snr_{subject_number}.csv"):
    """
    Load source-level SNR data and merge with subject-specific electrode-level SNR data.
    """
    all_dfs = []
    cwd = getcwd()

    for method in methods:
        for dist in dists:
            filename = source_file_template.format(method=method, dist=dist)

            for subj in p.subjects:
                subj_number = ''.join(filter(str.isdigit, subj))

                # Load source-level file
                source_path = path.join(cwd, p.data, p.processed_data,
                                        p.protocol, subj, p.session,
                                        p.current_dir, filename)

                if not path.exists(source_path):
                    print(f"Missing source file: {source_path}")
                    continue

                try:
                    df_source = pd.read_csv(source_path)
                    df_source["method"] = method
                    df_source["distance"] = dist
                    df_source["subject"] = subj_number
                except Exception as e:
                    print(f"Error loading source file {source_path}: {e}")
                    continue

                # Load corresponding electrode-level file
                electrode_filename = electrode_file_template.format(subject_number=subj_number)
                electrode_path = path.join(cwd, p.data, p.processed_data,
                                           p.protocol, subj, p.session,
                                           p.current_dir, electrode_filename)

                if not path.exists(electrode_path):
                    print(f"Missing electrode file: {electrode_path}")
                    continue

                try:
                    df_electrode = pd.read_csv(electrode_path)
                    df_electrode["subject"] = subj_number
                except Exception as e:
                    print(f"Error loading electrode file {electrode_path}: {e}")
                    continue


                # Rename 'n_select' in electrode to 'rep' temporarily just for merging
                df_electrode_renamed = df_electrode.rename(columns={"n_select": "rep"})
                
                # Then average across channels per subject + rep
                df_electrode_avg = (
                    df_electrode_renamed
                    .groupby(["subject", "rep"], as_index=False)
                    .agg(snr_mean_electrode=("snr", "mean"))
                )
                
                # Merge source (which has 'rep') with electrode (now renamed rep)
                merged = pd.merge(
                    df_source,
                    df_electrode_avg,
                    how="left",
                    on=["subject", "rep"]
                )

                all_dfs.append(merged)

    if not all_dfs:
        raise ValueError("No valid merged SNR data found.")

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Cast types
    full_df["method"] = full_df["method"].astype("category")
    full_df["distance"] = full_df["distance"].astype("category")
    full_df["subject"] = full_df["subject"].astype("category")
    full_df["rep"] = pd.to_numeric(full_df["rep"], errors="coerce")

    return full_df


def fit_lme_snr_model(snr_df, filter_s1_only=True):
    """
    Fit Linear Mixed Model: snr ~ method + distance + rep + (1|subject)
    Perform residual diagnostics (normality, homoscedasticity).
    """
    
    # ---------------------------------------------------
    # Filter for S1 ROIs if requested
    # ---------------------------------------------------
    if filter_s1_only:
        s1_patterns = [
            'L_3b_ROI-lh', 'L_1_ROI-lh', 'L_2_ROI-lh',
            'R_3b_ROI-rh', 'R_1_ROI-rh', 'R_2_ROI-rh',
            'L_4_ROI-lh', 'R_4_ROI-rh',
            'L_6d_ROI-lh', 'R_6d_ROI-rh',
            'L_6v_ROI-lh', 'R_6v_ROI-rh',
            'L_6a_ROI-lh', 'R_6a_ROI-rh',
            'L_6mp_ROI-lh', 'R_6mp_ROI-rh',
            'L_5m_ROI-lh', 'R_5m_ROI-rh',
            'L_5L_ROI-lh', 'R_5L_ROI-rh'
        ]
        s1_mask = snr_df['roi'].str.contains('|'.join(s1_patterns), case=False, na=False)
        snr_df = snr_df[s1_mask].copy()
        print(f"Filtered down to {snr_df.shape[0]} rows (S1 ROIs only)")
    
    # ---------------------------------------------------
    # Fit mixed model
    # ---------------------------------------------------
    snr_df['snr_db'] = np.log(snr_df['snr'])
    model = smf.mixedlm("snr_db ~ method + distance + rep", data=snr_df, groups=snr_df["subject"])
    result = model.fit()
    print(result.summary())
    
    # ---------------------------------------------------
    # Residuals
    # ---------------------------------------------------
    residuals = result.resid
    fitted = result.fittedvalues
    
    # ---------------------------------------------------
    # Normality tests
    # ---------------------------------------------------
    ks_stat, ks_p = kstest(residuals, "norm", args=(residuals.mean(), residuals.std()))
    print("\nKolmogorov-Smirnov test:")
    print(f"  KS statistic = {ks_stat:.4f}, p = {ks_p:.4f}")
    
    sh_stat, sh_p = shapiro(residuals)
    print("Shapiro-Wilk test:")
    print(f"  W = {sh_stat:.4f}, p = {sh_p:.4f}")
    
    ad_result = anderson(residuals)
    print("Anderson-Darling test:")
    print(f"  A^2 = {ad_result.statistic:.4f}")
    print(f"  Critical values = {ad_result.critical_values}, Significance levels = {ad_result.significance_level}")
    
    # ---------------------------------------------------
    # Homoscedasticity tests
    # ---------------------------------------------------
    groups = [residuals[snr_df["method"] == g] for g in snr_df["method"].unique()]
    lev_stat, lev_p = levene(*groups)
    print("\nLevene's test (by method):")
    print(f"  Statistic = {lev_stat:.4f}, p = {lev_p:.4f}")
    
    bp_test = het_breuschpagan(residuals, result.model.exog)
    print("Breusch-Pagan test:")
    print(f"  LM stat = {bp_test[0]:.4f}, p = {bp_test[1]:.4f}")
    
    white_test = het_white(residuals, result.model.exog)
    print("White's test:")
    print(f"  Stat = {white_test[0]:.4f}, p = {white_test[1]:.4f}")
    
    # ---------------------------------------------------
    # Multiple comparison correction
    # ---------------------------------------------------
    fixed_effects = result.pvalues[result.pvalues.notna()].drop("Intercept", errors="ignore")
    reject, pvals, _,_ = multipletests(fixed_effects, alpha=0.05, method="fdr_bh")
    
    corrected_df = pd.DataFrame({
        "coefficient": result.params[fixed_effects.index],
        "pval_raw": fixed_effects,
        "pval": pvals,
        "significant": reject
    })
    
    print("\nMultiple comparison correction (FDR):")
    print(corrected_df)

    tmp_df = snr_df.copy()
    tmp_df["resid"] = residuals  # residuals from your fitted model
    sh_pvals, lev_pvals = subsample_test_pvalues(tmp_df, "resid", group_col="method",
                                                 n_reps=200, sample_size=400, random_state=42)
    print("Subsample Shapiro p-values: median = {:.3g}, % > 0.05 = {:.1%}".format(np.nanmedian(sh_pvals), np.nanmean(sh_pvals > 0.05)))
    print("Subsample Levene p-values:   median = {:.3g}, % > 0.05 = {:.1%}".format(np.nanmedian(lev_pvals), np.nanmean(lev_pvals > 0.05)))
        
    # ---------------------------------------------------
    # Diagnostic plots
    # ---------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sm.qqplot(residuals, line='45', ax=axes[0,0])
    axes[0,0].set_title("QQ Plot of Residuals")
    
    sns.histplot(residuals, kde=True, ax=axes[0,1])
    axes[0,1].set_title("Histogram of Residuals")
    
    axes[1,0].scatter(fitted, residuals, alpha=0.6)
    axes[1,0].axhline(0, color='red', linestyle='--')
    axes[1,0].set_xlabel("Fitted values")
    axes[1,0].set_ylabel("Residuals")
    axes[1,0].set_title("Residuals vs. Fitted")
    
    sns.boxplot(x=snr_df["method"], y=residuals, ax=axes[1,1])
    axes[1,1].set_title("Residuals by Method")
    
    plt.tight_layout()
    plt.show()
    
    return result, corrected_df


def subsample_test_pvalues(df, resid_col, group_col="method", 
                           n_reps=100, sample_size=10000, random_state=0,
                           plot=True):
    """
    Draw repeated random subsamples of rows, run Shapiro and Levene,
    return p-value distributions and optionally plot histograms.
    """
    rng = np.random.RandomState(random_state)
    sh_p = []
    lev_p = []
    methods = df[group_col].unique()

    for i in range(n_reps):
        subs = df.sample(n=sample_size, replace=False, 
                         random_state=rng.randint(0, 1000000))
        resid_sub = subs[resid_col].values
        
        # Shapiro
        try:
            _, p_sh = shapiro(resid_sub)
        except Exception:
            p_sh = np.nan
        
        # Levene
        groups = [resid_sub[subs[group_col].values == m] for m in methods]
        try:
            _, p_lev = levene(*groups)
        except Exception:
            p_lev = np.nan
        
        sh_p.append(p_sh)
        lev_p.append(p_lev)

    sh_p = np.array(sh_p)
    lev_p = np.array(lev_p)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(sh_p, bins=20, color="steelblue", edgecolor="black")
        axes[0].axvline(0.05, color="red", linestyle="--")
        axes[0].set_title("Shapiro–Wilk p-values")
        axes[0].set_xlabel("p-value")
        axes[0].set_ylabel("Frequency")

        axes[1].hist(lev_p, bins=20, color="seagreen", edgecolor="black")
        axes[1].axvline(0.05, color="red", linestyle="--")
        axes[1].set_title("Levene's test p-values")
        axes[1].set_xlabel("p-value")

        plt.tight_layout()
        plt.show()

    return sh_p, lev_p


#----------------------------------------------------------------------------------#
def measure_entropy_el(p, output_csv="entropy_01.csv"):
    print("Measure entropy on electrode level: started")
    cwd = getcwd()
    processedeegpath = path.join(cwd, p.data, p.processed_data, 
                                 p.protocol, p.subj, p.session, 
                                 p.eeg_dir)
    
    fullpath_epochs = path.join(processedeegpath, p.epochsfile)
    epochs = mne.read_epochs(fullpath_epochs)
    selected_epochs = epochs[p.selected_epoch]

    n_epochs = len(selected_epochs)
    n_select = [125, 250, 500, 1000, 1500, 2000]
    n_repeats = 20

    data_list = []
    np.random.seed(11)

    def compute_entropy(signal):
        """Compute Shannon entropy of a 1D signal."""
        # Normalize the signal to form a probability distribution
        hist, bin_edges = np.histogram(signal, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero entries
        return shannon_entropy(hist)

    for n in n_select:
        entropy_values_per_channel = {ch: [] for ch in selected_epochs.ch_names}
        
        for _ in range(n_repeats):
            random_indices = np.random.choice(n_epochs, n, replace=False)
            random_subset = selected_epochs[random_indices]

            erp = random_subset.average()

            for ch_idx, ch_name in enumerate(selected_epochs.ch_names):
                signal = erp.data[ch_idx]  # 1D array for this channel
                ent = compute_entropy(signal)
                entropy_values_per_channel[ch_name].append(ent)

        for ch_name, ent_values in entropy_values_per_channel.items():
            data_list.append({
                "n_select": n,
                "channel": ch_name,
                "entropy_mean": np.mean(ent_values),
                "entropy_std": np.std(ent_values)
            })

    df = pd.DataFrame(data_list)

    output_path = path.join(cwd, p.data, p.processed_data, 
                            p.protocol, p.subj, p.session, 
                            p.current_dir, output_csv)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print("Measure entropy on electrode level: completed")


def plot_entropy_el_multi_subject(p):
    print("Plotting entropy across multiple subjects (Seaborn): started")
    cwd = getcwd()
    subjects = p.subjects
    channels_of_interest = ["C3", "Cz", "C4"]

    all_data = []

    for subj in subjects:
        subj_number = ''.join(filter(str.isdigit, subj))  # extract number if needed
        filename = f"snr_{subj_number}_entropy_el.csv"
        filepath = path.join(cwd, p.data, p.processed_data, 
                             p.protocol, subj, p.session, 
                             p.current_dir, filename)

        if not path.exists(filepath):
            print(f"Missing entropy file for {subj}. Skipping.")
            continue

        df = pd.read_csv(filepath)
        df = df[df["channel"].isin(channels_of_interest)].copy()
        df["Subject"] = subj
        all_data.append(df)

    if not all_data:
        print("No valid entropy data loaded. Exiting.")
        return

    df_all = pd.concat(all_data)

    # --- Plot Setup ---
    sns.set(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for i, ch in enumerate(channels_of_interest):
        ax = axes[i]
        df_ch = df_all[df_all["channel"] == ch]

        # Plot individual subjects in light grey
        sns.lineplot(
            data=df_ch,
            x="n_select", y="entropy_mean",
            hue="Subject", units="Subject",
            estimator=None, lw=1.2, alpha=0.4,
            color="lightgrey", legend=False, ax=ax
        )

        # Plot group average ± std in black
        sns.lineplot(
            data=df_ch,
            x="n_select", y="entropy_mean",
            estimator="mean", errorbar="sd",
            color="black", lw=2.5,
            label="Group Avg ± Std", ax=ax
        )

        ax.set_title(f"{ch}", fontsize=20, weight="bold")
        ax.set_xlabel("Repetitions", fontsize=13)
        if i == 0:
            ax.set_ylabel("Entropy", fontsize=13)
        else:
            ax.set_ylabel("")

        ax.grid(True, linestyle='--', alpha=0.5)

    fig.suptitle("Entropy for C3, Cz, C4 — Individual Subjects & Group Average", fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Shared legend in upper left
    custom_lines = [
        Line2D([0], [0], color='lightgrey', lw=1.2, alpha=0.4, label='Individual Subjects (N=11)'),
        Line2D([0], [0], color='black', lw=2.5, label='Group Avg ± Std')
    ]
    fig.legend(
        handles=custom_lines,
        loc="upper left", bbox_to_anchor=(0.01, 0.99),
        fontsize=12
    )

    plt.show()
    print("Plotting entropy across multiple subjects (Seaborn): completed")



def plot_entropy_from_csv(p, csv_file="entropy_01.csv", normalize="None", reference_channel=None, hist_n_select=None):
    cwd = getcwd()
    loadpath = path.join(cwd, p.data, p.processed_data, 
                         p.protocol, p.subj, p.session, 
                         p.current_dir, csv_file)
    df = pd.read_csv(loadpath)

    selected_electrodes = ['C5','C3','C1','Cz','C2','C4','C6']
    electrode_colors = {
        'C5': '#CC79A7',
        'C3': '#D55E00',
        'C6': '#0072B2',
        'Cz': '#F0E442',
        'C2': '#009E73',
        'C4': '#56B4E9',
        'C1': '#E69F00',
    }

    df_selected = df.query("channel in @selected_electrodes").copy()

    if normalize == "per_channel":
        df["entropy_mean"] = df.groupby("channel")["entropy_mean"].transform(lambda x: x / x.max())
        df["entropy_std"] = df.groupby("channel")["entropy_std"].transform(lambda x: x / x.max())
        df_selected = df.query("channel in @selected_electrodes").copy()

    elif normalize == "global" and reference_channel:
        ref_max = df[df["channel"] == reference_channel]["entropy_mean"].max()
        if pd.notna(ref_max) and ref_max > 0:
            df["entropy_mean"] = df["entropy_mean"] / ref_max
            df["entropy_std"] = df["entropy_std"] / ref_max
            df_selected = df.query("channel in @selected_electrodes").copy()

    # === LINE PLOT FOR SELECTED ELECTRODES ===
    mpl.rcParams.update({'font.size': 20})
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_selected, 
                 x="n_select", 
                 y="entropy_mean",
                 hue='channel', 
                 marker="o",
                 palette=electrode_colors)

    for electrode in selected_electrodes:
        df_electrode = df_selected[df_selected["channel"] == electrode]
        plt.fill_between(
            df_electrode["n_select"], 
            df_electrode["entropy_mean"] - df_electrode["entropy_std"], 
            df_electrode["entropy_mean"] + df_electrode["entropy_std"], 
            alpha=0.1,
            color=electrode_colors[electrode]
        )

    plt.xlabel("Number of Repetitions")
    plt.ylabel("Mean Entropy")
    plt.title("Participant 1")

    handles, labels = plt.gca().get_legend_handles_labels()
    desired_order = ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6']
    sorted_handles_labels = [(h, l) for l, h in zip(labels, handles) if l in desired_order]
    sorted_handles_labels.sort(key=lambda x: desired_order.index(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    plt.legend(handles=sorted_handles, labels=sorted_labels, title="Electrode")
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === HISTOGRAM FOR ALL ELECTRODES, SPECIFIC n_select ===
    if hist_n_select is not None:
        df_hist = df[df["n_select"] == hist_n_select]
        if df_hist.empty:
            print(f"No data found for n_select = {hist_n_select}. Histogram skipped.")
        else:
            plt.figure(figsize=(8, 5))
            sns.set(style="whitegrid")
            sns.histplot(df_hist["entropy_mean"], bins=30, kde=True, color="darkcyan", edgecolor="black")

            plt.xlabel("Mean Entropy", fontsize=16)
            plt.ylabel("Number of Electrodes", fontsize=16)
            plt.title(f"Histogram of Mean Entropy Across All Electrodes\n(n_select = {hist_n_select})", fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 6))
            sns.boxplot(x="n_select", y="entropy_mean", data=df_hist, color="lightgray")
            sns.swarmplot(x="n_select", y="entropy_mean", data=df_hist, color="teal", alpha=0.7)

            plt.ylabel("Mean Entropy")
            plt.xlabel("")
            plt.title(f"Entropy Distribution Across All Electrodes (n_select = {hist_n_select})")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
#------------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------------
# Additional debugging function
def debug_snr_data(p, subj):
    """Standalone function to debug SNR data issues"""
    
    subj_number = ''.join(filter(str.isdigit, subj))
    cwd = getcwd()
    
    # Load files
    snr_file = f"snr_{subj_number}_el.csv"
    snr_path = path.join(cwd, p.data, p.processed_data, p.protocol, subj, p.session, p.current_dir, snr_file)
    erp_path = path.join(cwd, p.data, p.processed_data, p.protocol, subj, p.session, p.eeg_dir, p.epochsfile)
    
    df = pd.read_csv(snr_path)
    epochs = mne.read_epochs(erp_path, preload=False)
    ch_names = epochs.info['ch_names']
    
    print(f"=== DETAILED DEBUG for {subj} ===")
    print(f"CSV shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"n_select values: {sorted(df['n_select'].unique())}")
    print(f"Channel count in CSV: {df['channel'].nunique()}")
    print(f"Channel count in ERP: {len(ch_names)}")
    
    # Check for channel mismatches
    csv_channels = set(df['channel'].unique())
    erp_channels = set(ch_names)
    
    print(f"\nChannels only in CSV: {csv_channels - erp_channels}")
    print(f"Channels only in ERP: {erp_channels - csv_channels}")
    
    # Check SNR statistics by repetition
    for rep in sorted(df['n_select'].unique()):
        rep_data = df[df['n_select'] == rep]
        snr_vals = rep_data['snr_mean'].dropna()
        print(f"\nRep {rep}:")
        print(f"  Rows: {len(rep_data)}")
        print(f"  SNR range: {snr_vals.min():.3f} to {snr_vals.max():.3f}")
        print(f"  SNR mean: {snr_vals.mean():.3f}")
        print(f"  NaN count: {rep_data['snr_mean'].isna().sum()}")
        
        # Show some example values
        print(f"  Sample values: {snr_vals.head().values}")
    
    return df, ch_names




def plot_custom_topomap(ax, values, ch_names, info, vmin=None, vmax=None, cmap='rocket', title=None):
    """
    Custom topomap with projected sensor positions into unit circle.
    """
    # Extract raw positions
    pos_raw = np.array([
        info['chs'][info.ch_names.index(ch)]['loc'][:2]
        for ch in ch_names if ch in info.ch_names
    ])
    vals = np.array([
        val for ch, val in zip(ch_names, values)
        if ch in info.ch_names
    ])

    # Normalize by max radius
    norm = np.linalg.norm(pos_raw, axis=1)

    # Extract raw positions (2D)
    pos_raw = np.array([
        info['chs'][info.ch_names.index(ch)]['loc'][:2]
        for ch in ch_names if ch in info.ch_names
    ])
    vals = np.array([
        val for ch, val in zip(ch_names, values)
        if ch in info.ch_names
    ])
    
    # --- Normalize sensor positions ---
    # Shift to mean-centered coordinates
    center = np.mean(pos_raw, axis=0)
    pos_centered = pos_raw - center
    
    # Scale so that max radius = 0.5
    radii = np.linalg.norm(pos_centered, axis=1)
    max_radius = np.max(radii)
    pos = (pos_centered / max_radius) * 0.5

    # Interpolation grid
    xi = yi = np.linspace(-0.6, 0.6, 300)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata(pos, vals, (xi, yi), method='cubic')

    # Apply circular mask
    mask = np.sqrt(xi**2 + yi**2) <= 0.5
    zi = np.ma.masked_where(~mask, zi)

    # Plot topomap
    im = ax.imshow(zi, origin='lower', extent=[-0.6, 0.6, -0.6, 0.6],
                   cmap=cmap, vmin=vmin, vmax=vmax)

    # Plot sensors
    ax.scatter(pos[:, 0], pos[:, 1], c='k', s=30, zorder=10)

    # Head circle
    #head = plt.Circle((0, 0), 0.5, edgecolor='k', facecolor='none', lw=1.5)
    #ax.add_patch(head)

    # Nose (inverted triangle)
    #nose = Polygon([[0, 0.55], [-0.05, 0.5], [0.05, 0.5]],
    #               closed=True, edgecolor='k', facecolor='none', lw=1.5)
    #ax.add_patch(nose)

    # Ears
    #left_ear = Ellipse((-0.5, 0), width=0.05, height=0.2, edgecolor='k', facecolor='none', lw=1.5)
    #right_ear = Ellipse((0.5, 0), width=0.05, height=0.2, edgecolor='k', facecolor='none', lw=1.5)
    #ax.add_patch(left_ear)
    #ax.add_patch(right_ear)

    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.6])
    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        ax.set_title(title)

    return im


#---------------------------------------------------------------------------------#
def analyze_electrode_source_snr(p, radius_mm=50):
    """
    Analyze SNR comparison between C3, Cz, C4 electrodes and nearby sources.
    
    Parameters:
    -----------
    p : parameter object
        Contains all the path and analysis parameters
    radius_mm : float
        Radius in millimeters to search for nearby sources (default: 50mm)
    
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame containing all electrode-source comparisons
    stats_results : dict
        Dictionary containing statistical test results for each electrode
    """
    print(f"Analyzing CP3, CPz, CP4 electrode-source SNR with {radius_mm}mm radius")
    
    cwd = getcwd()
    subjects = p.subjects
    rep = 2000  # Using single repetition value
    target_electrodes = ['CP3','CPz', 'CP4']
    radius_m = radius_mm / 1000.0  # Convert to meters
    
    # Store all data for analysis
    all_data = []
    # Store spatial data for 3D visualization
    spatial_data = {}
    
    for subj in subjects:
        print(f"\n{'='*50}")
        print(f"Processing subject: {subj}")
        print(f"{'='*50}")
        
        subj_number = ''.join(filter(str.isdigit, subj))

        # --- File Paths ---
        snr_file = f"snr_{subj_number}_el.csv"
        snr_path = path.join(cwd, p.data, p.processed_data, p.protocol, subj, p.session, p.current_dir, snr_file)
        erp_path = path.join(cwd, p.data, p.processed_data, p.protocol, subj, p.session, p.eeg_dir, p.epochsfile)
        stc_file = f"current{rep}reps_5mm_eLORETA-lh.stc"
        stc_path = path.join(cwd, p.data, p.processed_data, p.protocol, subj, p.session, p.current_dir, stc_file)

        # --- Check files exist ---
        if not all(path.exists(f) for f in [snr_path, erp_path, stc_path]):
            missing = [f for f in [snr_path, erp_path, stc_path] if not path.exists(f)]
            print(f"Missing files for {subj}: {missing}")
            continue

        try:
            # --- Load data ---
            epochs = mne.read_epochs(erp_path, preload=False)
            erp_info = epochs.info
            electrode_df = pd.read_csv(snr_path)
            electrode_data = electrode_df[electrode_df["n_select"] == rep]
            
            # Load STC and compute SNR
            stc = mne.read_source_estimate(stc_path)
            source_snr = snr_hammerer(p, stc.data, stc.times)
            
            print(f"Loaded data for {subj}: {len(electrode_data)} electrodes, {len(source_snr)} sources")

            # --- Load source space coordinates ---
            src_file_candidates = [
                path.join(cwd, p.data, p.processed_data, p.protocol, subj, p.session, p.eeg_dir, '5mm-src.fif'),
                path.join(cwd, p.data, p.processed_data, p.protocol, subj, p.session, p.eeg_dir, 'space_5mm-src.fif'),
                path.join(cwd, p.data, p.processed_data, p.protocol, subj, p.session, p.eeg_dir, 'ico-5-src.fif'),
            ]
            
            src_file = None
            for candidate in src_file_candidates:
                if path.exists(candidate):
                    src_file = candidate
                    break
            
            if src_file is None:
                print(f"Could not find source space file for {subj}")
                continue
                
            src_space = mne.read_source_spaces(src_file)
            
            # Get active source coordinates in head coordinate system
            lh_pos = src_space[0]['rr']
            rh_pos = src_space[1]['rr']
            lh_inuse = src_space[0]['inuse'].astype(bool)
            rh_inuse = src_space[1]['inuse'].astype(bool)
            
            all_pos = np.vstack([lh_pos, rh_pos])
            all_inuse = np.concatenate([lh_inuse, rh_inuse])
            active_source_coords = all_pos[all_inuse]
            
            # DEBUG: Print coordinate system info
            print(f"DEBUG: Total source positions: {len(all_pos)}")
            print(f"DEBUG: Active sources: {np.sum(all_inuse)}")
            print(f"DEBUG: Active source coords shape: {active_source_coords.shape}")
            print(f"DEBUG: Source coordinate ranges:")
            print(f"  X: [{active_source_coords[:, 0].min():.3f}, {active_source_coords[:, 0].max():.3f}]")
            print(f"  Y: [{active_source_coords[:, 1].min():.3f}, {active_source_coords[:, 1].max():.3f}]")
            print(f"  Z: [{active_source_coords[:, 2].min():.3f}, {active_source_coords[:, 2].max():.3f}]")
            
            # Match SNR data to active sources
            if len(source_snr) == len(active_source_coords):
                active_source_snr = source_snr
                print(f"DEBUG: SNR data matches active sources perfectly ({len(source_snr)})")
            elif len(source_snr) == len(all_inuse):
                active_source_snr = source_snr[all_inuse]
                print(f"DEBUG: SNR data matches all sources, filtered to active ({len(source_snr)} -> {len(active_source_snr)})")
            else:
                print(f"WARNING: SNR data length mismatch for {subj}")
                print(f"  SNR data: {len(source_snr)}, Active sources: {len(active_source_coords)}, All sources: {len(all_inuse)}")
                active_source_snr = source_snr[:len(active_source_coords)]

            # --- Get electrode positions using standard montage ---
            montage = mne.channels.make_standard_montage("standard_1005")
            erp_info_copy = erp_info.copy()
            erp_info_copy.set_montage(montage, match_case=False, on_missing='ignore')
            
            # Create KDTree for source lookup
            from scipy.spatial import cKDTree
            tree = cKDTree(active_source_coords)
            
            # Store electrode positions for 3D visualization
            electrode_positions = {}
            
            # --- Process target electrodes ---
            for electrode in target_electrodes:
                print(f"\n--- Processing electrode {electrode} ---")
                
                # Get electrode SNR
                elec_snr_row = electrode_data[electrode_data["channel"] == electrode]
                if elec_snr_row.empty:
                    print(f"No SNR data found for electrode {electrode} in {subj}")
                    continue
                    
                electrode_snr = elec_snr_row["snr_mean"].iloc[0]
                
                # Get electrode position
                try:
                    ch_idx = erp_info_copy['ch_names'].index(electrode)
                    electrode_pos = np.array(erp_info_copy['chs'][ch_idx]['loc'][:3])

                    if electrode==target_electrodes[0]:
                        electrode_pos[2] -= 0.06
                      #  electrode_pos[1] -= 0.035
                       # electrode_pos[0] += 0.05
                        pass
                         
                    elif electrode==target_electrodes[1]:
                        electrode_pos[2] -=0.06
                        pass
                    else:
                        electrode_pos[2] -= 0.06
                      #  electrode_pos[1] -= 0.015
                        pass
                   
                    # Check if position is valid
                    if np.allclose(electrode_pos, 0):
                        print(f"Invalid position for {electrode} in {subj}")
                        continue
                    
                    # DEBUG: Print electrode position info
                    print(f"DEBUG: {electrode} position: [{electrode_pos[0]:.3f}, {electrode_pos[1]:.3f}, {electrode_pos[2]:.3f}]")
                    print(f"DEBUG: {electrode} SNR: {electrode_snr:.2f}")
                    
                    electrode_positions[electrode] = electrode_pos
                        
                except (ValueError, IndexError):
                    print(f"Could not find electrode {electrode} in {subj}")
                    continue
                
                # Find sources within radius using KDTree
                nearby_indices = tree.query_ball_point(electrode_pos, r=radius_m)
                
                # DEBUG: Manual distance verification for first few sources
                if nearby_indices:
                    manual_distances = np.linalg.norm(active_source_coords - electrode_pos, axis=1)
                    manual_nearby = np.where(manual_distances <= radius_m)[0]
                    
                    print(f"DEBUG: KDTree found {len(nearby_indices)} sources within {radius_mm}mm")
                    print(f"DEBUG: Manual calculation found {len(manual_nearby)} sources within {radius_mm}mm")
                    
                    if len(nearby_indices) != len(manual_nearby):
                        print(f"WARNING: Mismatch between KDTree and manual calculation!")
                        print(f"  KDTree indices (first 10): {sorted(nearby_indices)[:10]}")
                        print(f"  Manual indices (first 10): {sorted(manual_nearby)[:10]}")
                        
                        # Show some distance examples
                        for i in range(min(5, len(active_source_coords))):
                            dist = manual_distances[i]
                            in_kdtree = i in nearby_indices
                            in_manual = i in manual_nearby
                            print(f"    Source {i}: dist={dist*1000:.1f}mm, KDTree={in_kdtree}, Manual={in_manual}")
                    
                    # Use manual calculation as backup if there's a mismatch
                    if len(nearby_indices) != len(manual_nearby):
                        print("Using manual calculation instead of KDTree")
                        nearby_indices = manual_nearby.tolist()
                
                if not nearby_indices:
                    print(f"No sources found within {radius_mm}mm of {electrode} for {subj}")
                    # DEBUG: Find closest source
                    closest_dist = np.min(np.linalg.norm(active_source_coords - electrode_pos, axis=1))
                    print(f"DEBUG: Closest source is {closest_dist*1000:.1f}mm away")
                    continue
                
                nearby_source_snr = active_source_snr[nearby_indices]
                n_sources = len(nearby_indices)
                
                # DEBUG: Show distance distribution
                nearby_distances = np.linalg.norm(active_source_coords[nearby_indices] - electrode_pos, axis=1)
                print(f"DEBUG: Distance range: {np.min(nearby_distances)*1000:.1f}-{np.max(nearby_distances)*1000:.1f}mm")
                print(f"DEBUG: Mean distance: {np.mean(nearby_distances)*1000:.1f}mm")
                
                print(f"{subj} - {electrode}: {n_sources} sources within {radius_mm}mm, "
                      f"electrode SNR: {electrode_snr:.2f}, "
                      f"source SNR range: {np.min(nearby_source_snr):.2f}-{np.max(nearby_source_snr):.2f}")
                
                # Store individual source data points
                for i, src_snr in enumerate(nearby_source_snr):
                    all_data.append({
                        'subject': subj,
                        'electrode': electrode,
                        'electrode_snr': electrode_snr,
                        'source_snr': src_snr,
                        'source_distance_mm': nearby_distances[i] * 1000,  # Add distance info
                        'n_sources_nearby': n_sources,
                        'radius_mm': radius_mm
                    })
            
            # Store spatial data for 3D visualization (use first subject with valid data)
            if not spatial_data and electrode_positions:
                spatial_data = {
                    'source_coords': active_source_coords,
                    'source_snr': active_source_snr,
                    'electrode_positions': electrode_positions,
                    'tree': tree,
                    'subject': subj
                }
                
        except Exception as e:
            print(f"Error processing {subj}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_data)
    
    if results_df.empty:
        print("No data collected! Check file paths and electrode names.")
        return results_df, {}
    
    print(f"\nCollected {len(results_df)} data points across {results_df['subject'].nunique()} subjects")
    print(f"Data per electrode: {results_df.groupby('electrode').size()}")
    
    # DEBUG: Show distance statistics
    if 'source_distance_mm' in results_df.columns:
        print(f"\nDEBUG: Distance statistics:")
        for electrode in target_electrodes:
            elec_data = results_df[results_df['electrode'] == electrode]
            if not elec_data.empty:
                distances = elec_data['source_distance_mm']
                print(f"  {electrode}: {len(distances)} sources, "
                      f"dist range: {distances.min():.1f}-{distances.max():.1f}mm, "
                      f"mean: {distances.mean():.1f}mm")
    
    # --- STATISTICAL ANALYSIS ---
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    
    from scipy import stats
    
    stats_results = {}
    
    for electrode in target_electrodes:
        print(f"\n--- Statistical tests for {electrode} ---")
        
        elec_data = results_df[results_df['electrode'] == electrode]
        if elec_data.empty:
            print(f"No data for {electrode}")
            continue
            
        electrode_snrs = elec_data['electrode_snr'].values
        source_snrs = elec_data['source_snr'].values
        
        # Calculate basic statistics
        elec_mean = np.mean(electrode_snrs)
        elec_std = np.std(electrode_snrs)
        source_mean = np.mean(source_snrs)
        source_std = np.std(source_snrs)
        
        print(f"Electrode SNR: {elec_mean:.3f} ± {elec_std:.3f} (n={len(electrode_snrs)})")
        print(f"Source SNR:    {source_mean:.3f} ± {source_std:.3f} (n={len(source_snrs)})")
        print(f"Difference:    {source_mean - elec_mean:.3f}")
        
        # Test 1: One-sample t-test (sources vs electrode mean)
        # Tests if source SNRs are significantly different from electrode SNR
        tstat, pval_ttest = stats.ttest_1samp(source_snrs, elec_mean)
        
        # Test 2: Wilcoxon signed-rank test (one-sample)
        # Non-parametric alternative to t-test
        # Tests if source SNRs have a median significantly different from electrode SNR
        diff_from_electrode = source_snrs - elec_mean
        wstat, pval_wilcoxon = wilcoxon(diff_from_electrode, alternative='two-sided')
        
        # Test 3: Mann-Whitney U test (if we want to compare distributions)
        # This compares electrode SNRs to source SNRs as two independent samples
        ustat, pval_mannwhitney = stats.mannwhitneyu(source_snrs, electrode_snrs, alternative='two-sided')
        
        # Test 4: Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(source_snrs) - 1) * source_std**2 + 
                             (len(electrode_snrs) - 1) * elec_std**2) / 
                             (len(source_snrs) + len(electrode_snrs) - 2))
        cohens_d = (source_mean - elec_mean) / pooled_std if pooled_std > 0 else 0
        
        # Test 5: Bootstrap confidence interval for mean difference
        n_bootstrap = 1000
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            bootstrap_sources = np.random.choice(source_snrs, len(source_snrs), replace=True)
            bootstrap_electrodes = np.random.choice(electrode_snrs, len(electrode_snrs), replace=True)
            bootstrap_diffs.append(np.mean(bootstrap_sources) - np.mean(bootstrap_electrodes))
        
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        # Store results
        stats_results[electrode] = {
            'electrode_mean': elec_mean,
            'electrode_std': elec_std,
            'electrode_n': len(electrode_snrs),
            'source_mean': source_mean,
            'source_std': source_std,
            'source_n': len(source_snrs),
            'mean_difference': source_mean - elec_mean,
            'ttest_statistic': tstat,
            'ttest_pvalue': pval_ttest,
            'wilcoxon_statistic': wstat,
            'wilcoxon_pvalue': pval_wilcoxon,
            'mannwhitney_statistic': ustat,
            'mannwhitney_pvalue': pval_mannwhitney,
            'cohens_d': cohens_d,
            'bootstrap_ci_lower': ci_lower,
            'bootstrap_ci_upper': ci_upper
        }
        
        # Print results
        print(f"\nStatistical Test Results:")
        print(f"  One-sample t-test:     t = {tstat:.3f}, p = {pval_ttest:.6f}")
        print(f"  Wilcoxon signed-rank:  W = {wstat:.3f}, p = {pval_wilcoxon:.6f}")
        print(f"  Mann-Whitney U:        U = {ustat:.3f}, p = {pval_mannwhitney:.6f}")
        print(f"  Cohen's d (effect size): {cohens_d:.3f}")
        print(f"  95% CI for mean diff: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        # Interpretation
        alpha = 0.05
        significant_tests = []
        if pval_ttest < alpha:
            significant_tests.append("t-test")
        if pval_wilcoxon < alpha:
            significant_tests.append("Wilcoxon")
        if pval_mannwhitney < alpha:
            significant_tests.append("Mann-Whitney")
        
        if significant_tests:
            direction = "higher" if source_mean > elec_mean else "lower"
            print(f"  → Sources have significantly {direction} SNR than electrode")
            print(f"    (significant in: {', '.join(significant_tests)})")
        else:
            print(f"  → No significant difference found")
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_size = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_size = "small"
        elif abs(cohens_d) < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        print(f"  → Effect size: {effect_size}")
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Electrode':<10} {'Elec SNR':<10} {'Source SNR':<12} {'Difference':<12} {'p-value':<12} {'Effect'}")
    print(f"{'-'*80}")
    
    for electrode in target_electrodes:
        if electrode in stats_results:
            r = stats_results[electrode]
            # Use Wilcoxon p-value as primary test (robust, non-parametric)
            significance = "*" if r['wilcoxon_pvalue'] < 0.05 else ""
            print(f"{electrode:<10} {r['electrode_mean']:<10.3f} {r['source_mean']:<12.3f} "
                  f"{r['mean_difference']:<12.3f} {r['wilcoxon_pvalue']:<12.6f} {r['cohens_d']:<.3f}{significance}")
    
    print(f"\n* p < 0.05 (Wilcoxon signed-rank test)")
    print(f"Effect size: Cohen's d (0.2=small, 0.5=medium, 0.8=large)")
    
    # --- Create visualizations ---
    create_electrode_source_plots(results_df, radius_mm, spatial_data=spatial_data)
    
    # --- Create 3D spatial visualization ---
    #if spatial_data:
    #    create_3d_spatial_plot(spatial_data, radius_mm, target_electrodes)
    
    return results_df, stats_results


# Additional debugging function to visualize the spatial relationship
def debug_electrode_source_distances(results_df, electrode='CP3', subject=None):
    """
    Debug function to examine distance distributions and potential issues.
    """
    import matplotlib.pyplot as plt
    
    if subject:
        data = results_df[(results_df['electrode'] == electrode) & (results_df['subject'] == subject)]
    else:
        data = results_df[results_df['electrode'] == electrode]
    
    if data.empty:
        print(f"No data found for electrode {electrode}")
        return
    
    distances = data['source_distance_mm']
    radius = data['radius_mm'].iloc[0]
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(distances, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(radius, color='red', linestyle='--', label=f'Radius: {radius}mm')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Number of sources')
    plt.title(f'{electrode} - Source Distance Distribution')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.scatter(distances, data['source_snr'], alpha=0.6)
    plt.axvline(radius, color='red', linestyle='--', label=f'Radius: {radius}mm')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Source SNR')
    plt.title(f'{electrode} - SNR vs Distance')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    # Check if sources are uniformly distributed by distance
    distance_bins = np.linspace(0, radius, 11)
    counts, _ = np.histogram(distances, bins=distance_bins)
    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
    plt.bar(bin_centers, counts, width=radius/10*0.8, alpha=0.7, edgecolor='black')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Number of sources')
    plt.title(f'{electrode} - Sources by Distance Bins')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nDistance statistics for {electrode}:")
    print(f"  Total sources: {len(distances)}")
    print(f"  Distance range: {distances.min():.1f} - {distances.max():.1f} mm")
    print(f"  Mean distance: {distances.mean():.1f} mm")
    print(f"  Std distance: {distances.std():.1f} mm")
    print(f"  Sources beyond radius: {np.sum(distances > radius)}")


def create_3d_spatial_plot(spatial_data, radius_mm, target_electrodes):
    """
    Create 3D plot showing electrodes and sources in space.
    Color sources based on which electrode's range they fall within.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    print(f"\nCreating 3D spatial visualization for subject {spatial_data['subject']}")
    
    # Extract data
    source_coords = spatial_data['source_coords']
    source_snr = spatial_data['source_snr']
    electrode_positions = spatial_data['electrode_positions']
    tree = spatial_data['tree']
    radius_m = radius_mm / 1000.0
    
    # Define colors for each electrode
    electrode_colors = {
        'CP3': '#FF6B6B',  # Red
        'CPz': '#4ECDC4',  # Teal
        'CP4': '#45B7D1'   # Blue
    }
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Create two subplots: one for electrode view, one for source SNR view
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # --- Plot 1: Color sources by electrode proximity ---
    ax1.set_title(f'Sources Colored by Electrode Proximity\n(radius: {radius_mm}mm)', fontsize=14, pad=20)
    
    # Initialize source colors (gray for sources not in any electrode's range)
    source_colors = ['lightgray'] * len(source_coords)
    source_labels = ['No electrode'] * len(source_coords)
    
    # For each electrode, find nearby sources and color them
    for electrode in target_electrodes:
        if electrode not in electrode_positions:
            continue
            
        electrode_pos = electrode_positions[electrode]
        nearby_indices = tree.query_ball_point(electrode_pos, r=radius_m)
        
        # Color sources within this electrode's range
        for idx in nearby_indices:
            if source_labels[idx] == 'No electrode':  # Only assign if not already assigned
                source_colors[idx] = electrode_colors[electrode]
                source_labels[idx] = electrode
            elif source_labels[idx] != electrode:  # Overlapping ranges
                source_colors[idx] = 'purple'  # Purple for overlapping
                source_labels[idx] = 'Multiple'
    
    # Plot sources
    x, y, z = source_coords.T
    ax1.scatter(x, y, z, c=source_colors, s=20, alpha=0.6)
    
    # Plot electrodes as larger spheres
    for electrode, pos in electrode_positions.items():
        ax1.scatter(pos[0], pos[1], pos[2], 
                   c=electrode_colors[electrode], s=200, 
                   marker='o', edgecolor='black', linewidth=2,
                   label=f'{electrode} electrode')
        
        # Draw sphere representing electrode range
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        sphere_x = radius_m * np.outer(np.cos(u), np.sin(v)) + pos[0]
        sphere_y = radius_m * np.outer(np.sin(u), np.sin(v)) + pos[1]
        sphere_z = radius_m * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
        ax1.plot_wireframe(sphere_x, sphere_y, sphere_z, 
                          color=electrode_colors[electrode], alpha=0.1, linewidth=0.5)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # --- Plot 2: Color sources by SNR values ---
    ax2.set_title('Sources Colored by SNR Values', fontsize=14, pad=20)
    
    # Create colormap for SNR values
    scatter = ax2.scatter(x, y, z, c=source_snr, s=20, alpha=0.7, 
                         cmap='viridis', vmin=np.percentile(source_snr, 5), 
                         vmax=np.percentile(source_snr, 95))
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8, pad=0.1)
    cbar.set_label('Source SNR', rotation=270, labelpad=15)
    
    # Plot electrodes
    for electrode, pos in electrode_positions.items():
        ax2.scatter(pos[0], pos[1], pos[2], 
                   c=electrode_colors[electrode], s=200, 
                   marker='o', edgecolor='black', linewidth=2,
                   label=f'{electrode} electrode')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set equal aspect ratio for both plots
    for ax in [ax1, ax2]:
        # Get the range of the data
        x_range = np.ptp(x)
        y_range = np.ptp(y)
        z_range = np.ptp(z)
        max_range = max(x_range, y_range, z_range)
        
        # Set limits to center the data
        x_center = np.mean(x)
        y_center = np.mean(y)
        z_center = np.mean(z)
        
        ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
        ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
    
    plt.tight_layout()
    plt.show()
    
    # --- Create a summary plot showing source counts ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count sources in each electrode's range
    electrode_source_counts = {}
    electrode_mean_snr = {}
    
    for electrode in target_electrodes:
        if electrode not in electrode_positions:
            continue
            
        electrode_pos = electrode_positions[electrode]
        nearby_indices = tree.query_ball_point(electrode_pos, r=radius_m)
        
        electrode_source_counts[electrode] = len(nearby_indices)
        if nearby_indices:
            electrode_mean_snr[electrode] = np.mean(source_snr[nearby_indices])
        else:
            electrode_mean_snr[electrode] = 0
    
    # Bar plot of source counts
    electrodes = list(electrode_source_counts.keys())
    counts = list(electrode_source_counts.values())
    colors = [electrode_colors[e] for e in electrodes]
    
    bars1 = ax1.bar(electrodes, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title(f'Number of Sources within {radius_mm}mm of Each Electrode')
    ax1.set_ylabel('Number of Sources')
    ax1.set(xlabel=None)
    
    # Add value labels on bars
    for bar, count in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Bar plot of mean SNR
    mean_snrs = list(electrode_mean_snr.values())
    bars2 = ax2.bar(electrodes, mean_snrs, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title(f'Mean SNR of Sources within {radius_mm}mm of Each Electrode')
    ax2.set_ylabel('Mean Source SNR')
    ax2.set(xlabel=None)
    
    # Add value labels on bars
    for bar, snr in zip(bars2, mean_snrs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{snr:.2f}', ha='center', va='bottom', fontweight='bold')

    
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n3D Spatial Analysis Summary:")
    print(f"Total sources: {len(source_coords)}")
    print(f"Source SNR range: {np.min(source_snr):.2f} - {np.max(source_snr):.2f}")
    
    for electrode in target_electrodes:
        if electrode in electrode_source_counts:
            print(f"{electrode}: {electrode_source_counts[electrode]} sources, "
                  f"mean SNR: {electrode_mean_snr[electrode]:.2f}")


def create_electrode_source_plots(df, radius_mm, spatial_data=None, top_sources_percent=10, exclude_subjects=None):
    """
    Create visualizations comparing electrode and source SNRs.
    Includes shared-scale boxplots and a top-down spatial map.
    
    Parameters:
    -----------
    df : DataFrame
        Data containing electrode and source SNR values
    radius_mm : float
        Radius in millimeters for electrode range
    spatial_data : dict, optional
        Spatial coordinates and electrode positions
    top_sources_percent : float, default=5
        Percentage of top sources to include in analysis (e.g., 5 for top 5%)
    exclude_subjects : list, optional
        List of subject IDs to exclude from analysis
    """
    
    if df.empty:
        print("No data to plot!")
        return
    
    # Filter out excluded subjects
    if exclude_subjects is not None:
        df = df[~df['subject'].isin(exclude_subjects)]
        print(f"Excluded subjects: {exclude_subjects}")
        if df.empty:
            print("No data remaining after subject exclusion!")
            return
    
    sns.set(style="whitegrid", font_scale=1.2)

    # Use consistent electrode order and colors
    target_electrodes = ['CP3', 'CPz', 'CP4']
    electrode_colors = {
        'CP3': '#FF6B6B',   # red
        'CPz': '#4ECDC4',   # teal
        'CP4': '#45B7D1'    # blue
    }

    # Prepare data
    df = df[df['electrode'].isin(target_electrodes)]
    df['electrode'] = pd.Categorical(df['electrode'], categories=target_electrodes, ordered=True)
    
    electrode_summary = df.groupby(['subject', 'electrode'])['electrode_snr'].first().reset_index()
    electrode_summary['electrode'] = pd.Categorical(electrode_summary['electrode'], categories=target_electrodes, ordered=True)

    # Calculate top sources for each electrode-subject combination (using custom percentage)
    extended_data = []
    
    for electrode in target_electrodes:
        for subject in df['subject'].unique():
            subset = df[(df['electrode'] == electrode) & (df['subject'] == subject)]
            
            if subset.empty:
                print(f"No data found for electrode '{electrode}' in subject '{subject}'")
                continue  # skip this subject and move on
            
            # Extract electrode SNR
            electrode_snr = subset['electrode_snr'].iloc[0]
            
            # All sources mean
            all_sources_mean = subset['source_snr'].mean()
            
            # Top X% sources (customizable)
            n_top = max(1, int(np.ceil(len(subset) * (top_sources_percent / 100))))
            top_sources = subset.nlargest(n_top, 'source_snr')
            top_mean = top_sources['source_snr'].mean()
            
            # Add data points for violin plot
            extended_data.extend([
                {'subject': subject, 'electrode': electrode, 'SNR_type': 'Electrode', 'SNR_value': electrode_snr},
                {'subject': subject, 'electrode': electrode, 'SNR_type': 'All Sources', 'SNR_value': all_sources_mean},
                {'subject': subject, 'electrode': electrode, 'SNR_type': f'Top {top_sources_percent}%', 'SNR_value': top_mean}
            ])

    extended_df = pd.DataFrame(extended_data)
    extended_df['electrode'] = pd.Categorical(extended_df['electrode'], categories=target_electrodes, ordered=True)

    # Setup figure
    plt.rcParams.update({
        'font.size': 18,        # Default text size
        'axes.labelsize': 18,   # Axis labels
        'axes.titlesize': 18,   # Subplot titles
        'xtick.labelsize': 18,  # X-axis tick labels
        'ytick.labelsize': 18,  # Y-axis tick labels
        'legend.fontsize': 14,  # Legend text
    })

    new_labels = ["3 cm to C3", "3 cm to Cz", "3 cm to C4"]
    el_labels = ["C3", "Cz", "C4"]
    
    # Create figure with 2x3 grid to accommodate the split brain subplot
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.2, wspace=0.5)
    
    # Define subplots
    ax_brain_3d      = fig.add_subplot(gs[0, 0])     # 3/4 view brain (original)
    ax_electrode_snr = fig.add_subplot(gs[0, 1:])     # Top view brain (new)
#    ax_source_snr    = fig.add_subplot(gs[0, 2:])    # Source SNRs (spans 2 columns)
#    ax_violin        = fig.add_subplot(gs[1, 0])     # Violin plot (swapped from (2,1))
    ax_brain_top     = fig.add_subplot(gs[1, 1])     # Electrode SNRs 
    ax_ratio         = fig.add_subplot(gs[1, 2:])  # SNR ratio violin plot


    # --- Synchronize Y-axis first ---
    all_snr = pd.concat([df['source_snr'], electrode_summary['electrode_snr'], extended_df['SNR_value']])
    y_min, y_max = all_snr.min(), all_snr.max()

    # --- Electrode SNRs (now on right) ---
    # --- Electrode and Source SNRs (paired connected scatter plot) ---
    # Calculate mean source SNR per subject per electrode
    source_summary = df.groupby(['subject', 'electrode'])['source_snr'].mean().reset_index()
    source_summary['electrode'] = pd.Categorical(source_summary['electrode'], categories=target_electrodes, ordered=True)
    
    # Create 6 columns: C3_elec, C3_source, Cz_elec, Cz_source, C4_elec, C4_source
    x_positions = {
        'CP3': {'electrode': 0, 'source': 1},
        'CPz': {'electrode': 2, 'source': 3},
        'CP4': {'electrode': 4, 'source': 5}
    }
    
    # Store data for statistical testing
    paired_data = {elec: {'electrode': [], 'source': []} for elec in target_electrodes}
    
    # Plot for each participant
    for subject in electrode_summary['subject'].unique():
        for elec in target_electrodes:
            # Get electrode SNR
            elec_data = electrode_summary[(electrode_summary['subject'] == subject) & 
                                         (electrode_summary['electrode'] == elec)]
            # Get source SNR
            source_data = source_summary[(source_summary['subject'] == subject) & 
                                        (source_summary['electrode'] == elec)]
            
            if not elec_data.empty and not source_data.empty:
                elec_snr = elec_data['electrode_snr'].values[0]
                source_snr = source_data['source_snr'].values[0]
                
                # Store for statistical testing
                paired_data[elec]['electrode'].append(elec_snr)
                paired_data[elec]['source'].append(source_snr)
                
                x_elec = x_positions[elec]['electrode']
                x_source = x_positions[elec]['source']
                
                # Draw connecting line
                ax_electrode_snr.plot([x_elec, x_source], [elec_snr, source_snr], 
                                     color='gray', alpha=0.3, linewidth=1, zorder=1)
                
                # Plot electrode diamond
                ax_electrode_snr.scatter(x_elec, elec_snr, marker='D', s=80,
                                        color=electrode_colors[elec],
                                        edgecolor='black', linewidth=0.5,
                                        alpha=0.7, zorder=2)
                
                # Plot source diamond
                ax_electrode_snr.scatter(x_source, source_snr, marker='D', s=80,
                                        color=electrode_colors[elec],
                                        edgecolor='black', linewidth=0.5,
                                        alpha=0.7, zorder=2)
    
    # Perform Wilcoxon signed-rank tests (two-tailed) and add significance markers
    print("\n" + "="*70)
    print("WILCOXON SIGNED-RANK TESTS: ELECTRODE vs SOURCE SNR (two-tailed)")
    print("="*70)
    
    for elec in target_electrodes:
        elec_values = np.array(paired_data[elec]['electrode'])
        source_values = np.array(paired_data[elec]['source'])
        
        if len(elec_values) > 0 and len(source_values) > 0:
            stat, pval = wilcoxon(elec_values, source_values, alternative='two-sided')
            
            # Determine significance
            if pval < 0.05:
                sig_label = '*'
            else:
                sig_label = 'n.s.'
            
            print(f"\n{elec}:")
            print(f"  Electrode: {np.mean(elec_values):.3f} ± {np.std(elec_values):.3f}")
            print(f"  Source:    {np.mean(source_values):.3f} ± {np.std(source_values):.3f}")
            print(f"  Wilcoxon:  W = {stat:.3f}, p = {pval:.4f} {sig_label}")
            
            # Add significance label to plot
            x_mid = (x_positions[elec]['electrode'] + x_positions[elec]['source']) / 2
            y_pos = y_max - 0.25 * (y_max - y_min)  # Position near top of plot
            
            ax_electrode_snr.text(x_mid, y_pos, sig_label, 
                                 ha='center', va='bottom',
                                 fontsize=20, fontweight='bold',
                                 color=electrode_colors[elec])
    
    ax_electrode_snr.set_title('')
    ax_electrode_snr.set_ylabel('SNR')
    ax_electrode_snr.set_xlabel("")
    ax_electrode_snr.set_xticks([0, 1, 2, 3, 4, 5])
    ax_electrode_snr.set_xticklabels(['CP3\nElec', 'CP3\nSource', 'CPz\nElec', 'CPz\nSource', 'CP4\nElec', 'CP4\nSource'])
    ax_electrode_snr.set_ylim(y_min, y_max)
    # --- Source SNRs (now spans bottom) ---
#    sns.boxplot(data=df, x='electrode', y='source_snr', ax=ax_source_snr,
#                palette=[electrode_colors[e] for e in target_electrodes], showfliers=False, whis=[5, 95], width=0.5)
#    sns.stripplot(data=df, x='electrode', y='source_snr', ax=ax_source_snr,
#                  hue='subject', dodge=True, jitter=0.25, marker='o', size=5,
#                  palette='gray', alpha=0.5, linewidth=0.5)
#    ax_source_snr.set_title('')
#    ax_source_snr.set_ylabel('SNR')
#    ax_source_snr.set_xlabel("")
#    ax_source_snr.set_xticklabels(new_labels)
#    ax_source_snr.legend_.remove()
#    ax_source_snr.set_ylim(y_min, y_max)
#    
    from mpl_toolkits.mplot3d import Axes3D

    def project_points_mpl(x, y, z, elev=30, azim=-60):
        """
        Project 3D points into 2D using Matplotlib's Axes3D projection.
        Works with both single points (scalars) and arrays.
        """
        # Create dummy 3D axis to get projection matrix
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        M = ax.get_proj()
        plt.close(fig)
    
        # Ensure inputs are arrays
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
    
        # Pack into homogeneous coords
        pts = np.stack([x, y, z], axis=-1)        # (N, 3)
        pts_hom = np.c_[pts, np.ones(len(pts))]   # (N, 4)
    
        # Apply projection
        proj = pts_hom @ M.T                      # (N, 4)
    
        # Normalize by w
        proj /= proj[:, 3][:, None]
    
        # Return 1D arrays (or scalars if input was scalar)
        xp, yp = proj[:, 0], proj[:, 1]
        if xp.size == 1:
            return xp.item(), yp.item()
        return xp, yp
  
    
    # --- Brain subplots (split into 3/4 view and top view) ---
    if spatial_data:
        source_coords = spatial_data['source_coords']
        electrode_positions = spatial_data['electrode_positions']
        tree = spatial_data['tree']
        radius_m = radius_mm / 1000.0
        source_snr_spatial = spatial_data['source_snr']
    
        source_colors = ['lightgray'] * len(source_coords)
        source_labels = ['None'] * len(source_coords)
        source_electrode_snrs = np.full(len(source_coords), np.nan)  # For electrode SNR mapping

        for elec in target_electrodes:
            if elec not in electrode_positions:
                continue
            pos = electrode_positions[elec]
            nearby_idx = tree.query_ball_point(pos, r=radius_m)
            
            # Get electrode SNR for this electrode (averaged across subjects)
            elec_snr = electrode_summary[electrode_summary['electrode'] == elec]['electrode_snr'].mean()
            
            for idx in nearby_idx:
                if source_labels[idx] == 'None':
                    source_colors[idx] = electrode_colors[elec]
                    source_labels[idx] = elec
                    source_electrode_snrs[idx] = elec_snr
                elif source_labels[idx] != elec:
                    source_colors[idx] = 'purple'
                    source_labels[idx] = 'Multiple'
                    # For multiple electrode overlap, use mean of electrode SNRs
                    prev_elec_snr = source_electrode_snrs[idx]
                    if not np.isnan(prev_elec_snr):
                        source_electrode_snrs[idx] = (prev_elec_snr + elec_snr) / 2
    
        x, y, z = source_coords.T
        
        # --- 3/4 view (original) ---
        x2d, y2d = project_points_mpl(x, y, z, elev=50, azim=-75)
        
        # Convert to numpy array
        source_colors_arr = np.array(source_colors, dtype=str)
        
        # Define gray vs colored masks
        gray_mask   = np.char.lower(source_colors_arr) == "lightgray"
        color_mask  = ~gray_mask
        
        # Plot gray sources in background
        ax_brain_3d.scatter(x2d[gray_mask], y2d[gray_mask],
                    c=source_colors_arr[gray_mask],
                    s=15, alpha=0.3, zorder=1)
        
        # Plot colored sources on top
        ax_brain_3d.scatter(x2d[color_mask], y2d[color_mask],
                    c=source_colors_arr[color_mask],
                    s=20, alpha=0.9, zorder=2)
        
        # Plot electrodes (always on top)
        for elec, pos in electrode_positions.items():
            px, py = project_points_mpl(*pos, elev=50, azim=-75)
            ax_brain_3d.scatter(px, py,
                        c=electrode_colors[elec], s=100,
                        edgecolor='black', zorder=3)
            ax_brain_3d.text(px + 0.001, py, elec,
                     ha='center', va='bottom',
                     fontsize=18, zorder=4)
        
        ax_brain_3d.set_aspect("equal")
        ax_brain_3d.set_xlabel("")
        ax_brain_3d.set_ylabel("")
        ax_brain_3d.set_xticks([])
        ax_brain_3d.set_yticks([])
        ax_brain_3d.axis("off")
        ax_brain_3d.set_title("3/4 View")
        
        # --- Top view (new) ---
        # Project from top (elev=90, azim=0 for pure top-down)
        x2d_top, y2d_top = project_points_mpl(x, y, z, elev=90, azim=-90)
        
        # Calculate source SNR relative to electrode SNR
        source_snr_ratio = np.full(len(source_coords), np.nan)
        valid_mask = ~np.isnan(source_electrode_snrs)
        source_snr_ratio[valid_mask] = source_snr_spatial[valid_mask] / source_electrode_snrs[valid_mask]
        
        # Plot sources with colormap based on SNR ratio
        valid_ratio_mask = ~np.isnan(source_snr_ratio)
        invalid_ratio_mask = np.isnan(source_snr_ratio)
        
        # Plot invalid (gray) sources
        ax_brain_top.scatter(x2d_top[invalid_ratio_mask], y2d_top[invalid_ratio_mask],
                           c='lightgray', s=15, alpha=0.3, zorder=1)
        
        # Plot valid sources with ratio colormap
        if np.sum(valid_ratio_mask) > 0:
            scatter = ax_brain_top.scatter(x2d_top[valid_ratio_mask], y2d_top[valid_ratio_mask],
                                         c=source_snr_ratio[valid_ratio_mask], s=20, alpha=0.8, 
                                         cmap='RdYlBu_r', vmin=np.min(source_snr_ratio[valid_ratio_mask]), 
                                           vmax=np.max(source_snr_ratio[valid_ratio_mask]), zorder=2)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax_brain_top, shrink=0.8)
            #cbar.set_label('Mean SNR Ratio', rotation=270, labelpad=15)
        
        ax_brain_top.set_aspect("equal")
        ax_brain_top.set_xlabel("")
        ax_brain_top.set_ylabel("")
        ax_brain_top.set_xticks([])
        ax_brain_top.set_yticks([])
        ax_brain_top.axis("off")
        #ax_brain_top.set_title("Mean SNR Ratio")
        
    else:
        ax_brain_3d.axis('off')
        ax_brain_3d.text(0.5, 0.5, "No spatial data", ha='center', va='center', transform=ax_brain_3d.transAxes)
        ax_brain_top.axis('off')
        ax_brain_top.text(0.5, 0.5, "No spatial data", ha='center', va='center', transform=ax_brain_top.transAxes)


    # --- Statistical testing and plot for SNR ratios ---
    if spatial_data and np.sum(valid_ratio_mask) > 0:
        # Prepare data for statistical testing
        ratio_test_data = []
        
        for elec in target_electrodes:
            if elec not in electrode_positions:
                continue
                
            pos = electrode_positions[elec]
            nearby_idx = tree.query_ball_point(pos, r=radius_m)
            
            # Get SNR ratios for sources in this electrode's range
            elec_mask = np.isin(np.arange(len(source_coords)), nearby_idx)
            valid_elec_mask = elec_mask & valid_ratio_mask
            
            if np.sum(valid_elec_mask) > 0:
                elec_ratios = source_snr_ratio[valid_elec_mask]
                
                # All sources ratios
                for subject in df['subject'].unique():
                    ratio_test_data.extend([
                        {'subject': subject, 'electrode': elec, 'ratio_type': 'All Sources', 'snr_ratio': np.mean(elec_ratios)},
                    ])
                
                # Top 10% sources ratios
                n_top = max(1, int(np.ceil(len(elec_ratios) * 0.1)))
                top_ratios = np.partition(elec_ratios, -n_top)[-n_top:]
                for subject in df['subject'].unique():
                    ratio_test_data.extend([
                        {'subject': subject, 'electrode': elec, 'ratio_type': 'Top 10%', 'snr_ratio': np.mean(top_ratios)},
                    ])
        
        # Create DataFrame for statistical testing
        ratio_df = pd.DataFrame(ratio_test_data)
        ratio_df['electrode'] = pd.Categorical(ratio_df['electrode'], categories=target_electrodes, ordered=True)

    # --- SNR Ratio Analysis Plot (ax_ratio) ---
    # Define electrode colors
    electrode_colors = {
        'CP3': '#FF6B6B',   # red
        'CPz': '#4ECDC4',   # teal
        'CP4': '#45B7D1'    # blue
    }
    
    if ratio_df.empty:
        ax_ratio.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax_ratio.transAxes)
    else:
        # Prepare data for individual source ratios (not averaged)
        individual_ratio_data = []
        
        for elec in target_electrodes:
            if elec not in electrode_positions:
                continue
                
            pos = electrode_positions[elec]
            nearby_idx = tree.query_ball_point(pos, r=radius_m)
            
            # Get SNR ratios for sources in this electrode's range
            elec_mask = np.isin(np.arange(len(source_coords)), nearby_idx)
            valid_elec_mask = elec_mask & valid_ratio_mask
            
            if np.sum(valid_elec_mask) > 0:
                elec_ratios = source_snr_ratio[valid_elec_mask]
                
                # Add each individual source ratio to the data
                for ratio in elec_ratios:
                    individual_ratio_data.append({
                        'electrode': elec, 
                        'snr_ratio': ratio,
                        'ratio_type': 'All'
                    })
                
                # Add top 10% sources
                n_top = max(1, int(np.ceil(len(elec_ratios) * 0.1)))
                top_ratios = np.partition(elec_ratios, -n_top)[-n_top:]
                for ratio in top_ratios:
                    individual_ratio_data.append({
                        'electrode': elec, 
                        'snr_ratio': ratio,
                        'ratio_type': 'Top 10%'
                    })
        
        # Create DataFrame with individual source ratios
        individual_df = pd.DataFrame(individual_ratio_data)
        individual_df['electrode'] = pd.Categorical(individual_df['electrode'], categories=target_electrodes, ordered=True)
        
        # Create vertical boxplots with electrode-specific colors
        # We need to create separate plots for each electrode to control colors
        box_positions = []
        box_colors = []
        
        # Get unique ratio types for positioning
        ratio_types = individual_df['ratio_type'].unique()
        
        for i, electrode in enumerate(target_electrodes):
            for j, ratio_type in enumerate(ratio_types):
                subset = individual_df[(individual_df['electrode'] == electrode) & 
                                     (individual_df['ratio_type'] == ratio_type)]
                if len(subset) > 0:
                    # Calculate position for this box
                    position = i * (len(ratio_types) + 0.5) + j
                    box_positions.append(position)
                    box_colors.append(electrode_colors[electrode])
                    
                    # Create individual boxplot
                    box_data = subset['snr_ratio'].values
                    bp = ax_ratio.boxplot(box_data, positions=[position], widths=0.4, 
                                        patch_artist=True, showfliers=False,
                                        whis=[5, 95])
                    for median in bp['medians']:
                        median.set_color('black')
                    
                    # Color the box
                    bp['boxes'][0].set_facecolor(electrode_colors[electrode])
                    bp['boxes'][0].set_alpha(0.7)
                    
                    # Add individual points
                    np.random.seed(42)  # For consistent jitter
                    x_jitter = np.random.normal(position, 0.05, len(box_data))
                    ax_ratio.scatter(x_jitter, box_data, alpha=0.5, s=15, 
                                   color='gray', zorder=3)
        
        # Add baseline line at SNR ratio = 1.0
        ax_ratio.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
        
        # --- Wilcoxon signed-rank test vs baseline ---
        print("="*70)
        print("SNR RATIO WILCOXON TESTS")
        print("="*70)
        
        # Get axis limits to position asterisks inside the plot
        ax_y_min, ax_y_max = ax_ratio.get_ylim()
        
        for i, electrode in enumerate(target_electrodes):
            for j, ratio_type in enumerate(ratio_types):
                subset = individual_df[(individual_df['electrode'] == electrode) & 
                                     (individual_df['ratio_type'] == ratio_type)]
                if len(subset) > 5:
                    data = subset['snr_ratio'].values
                    stat, p_value = wilcoxon(data - 1.0, alternative='greater')
                    
                    print(f"{electrode} ({ratio_type}): median={np.median(data):.3f}, "
                          f"p={p_value:.4f} {'✓' if p_value < 0.05 else '✗'}")
                    
                    # Add significance asterisk if p < 0.05, positioned inside plot
                    if p_value < 0.05:
                        # Position asterisk at 90% of the plot height
                        y_pos = ax_y_min + 0.6 * (ax_y_max - ax_y_min)
                        x_pos = i * (len(ratio_types) + 0.5) + j
                        ax_ratio.text(x_pos, y_pos, '*', ha='center', va='center',
                                    fontsize=32, fontweight='bold',
                                    color=electrode_colors[electrode])
        
        # Formatting
        ax_ratio.set_ylabel("Mean SNR Ratio", fontsize=18)
        ax_ratio.set_xlabel("", fontsize=12)
        
        # Set x-axis labels and ticks
        x_labels = []
        x_positions = []
        for i, electrode in enumerate(target_electrodes):
            for j, ratio_type in enumerate(ratio_types):
                x_positions.append(i * (len(ratio_types) + 0.5) + j)
                x_labels.append(f"{ratio_type}")
        
        ax_ratio.set_xticks(x_positions)
        ax_ratio.set_xticklabels(x_labels, fontsize=18)
        
        ax_ratio.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax_ratio.set_axisbelow(True)
        
        # Add a legend for electrode colors
        #legend_elements = [plt.Rectangle((0,0),1,1, facecolor=electrode_colors[electrode], 
        #                               alpha=0.7, label=electrode) for electrode in target_electrodes]
        #ax_ratio.legend(handles=legend_elements, title="Electrode", 
        #               loc="upper right", framealpha=0.9)
                    
    # --- Violin plot (now on bottom left) ---
    #sns.violinplot(data=extended_df, x='electrode', y='SNR_value', hue='SNR_type', ax=ax_violin,
    #               palette=['#dfdfdf', '#c0c0c0', '#808080'], inner=None)
    #sns.stripplot(data=extended_df, x='electrode', y='SNR_value', hue='SNR_type',
    #              dodge=True, jitter=True, marker='o', alpha=0.6, size=4,
    #              ax=ax_violin, palette=['#404040', '#606060', '#202020'], linewidth=0.5)
    #
    #ax_violin.set_ylabel('SNR')
    #ax_violin.set_xlabel(None)
    #ax_violin.set_xticklabels(new_labels)
    #ax_violin.set_ylim(y_min, y_max)
    #
    #handles, labels = ax_violin.get_legend_handles_labels()
    #ax_violin.legend(handles[:3], ['Electrode', 'All Sources', f'Top {top_sources_percent}%'], title='SNR Distribution')


    # --- Perform Wilcoxon signed-rank tests with FDR correction ---
    print(f"\n{'='*80}")
    print("WILCOXON SIGNED-RANK TESTS: ALL SOURCES vs TOP 10%")
    print(f"{'='*80}")
    
    # Store all test results for FDR correction
    all_pvalues = []
    all_stats = []
    test_labels = []
    electrode_results = {}
    
    for electrode in target_electrodes:
        elec_data = extended_df[extended_df['electrode'] == electrode]
        
        all_sources_data = elec_data[elec_data['SNR_type'] == 'All Sources']['SNR_value'].values
        top5_data = elec_data[elec_data['SNR_type'] == 'Top 10%']['SNR_value'].values
        electrode_data = elec_data[elec_data['SNR_type'] == 'Electrode']['SNR_value'].values
        
        if len(all_sources_data) > 0 and len(top5_data) > 0:
            # Test: All Sources vs Top 10%
            diff_all_vs_top5 = top5_data - all_sources_data
            if len(diff_all_vs_top5) > 0:
                stat1, pval1 = wilcoxon(diff_all_vs_top5, alternative='greater')
                
                # Test: Electrode vs All Sources
                diff_elec_vs_all = all_sources_data - electrode_data
                stat2, pval2 = wilcoxon(diff_elec_vs_all, alternative='greater')
                
                # Test: Electrode vs Top 10%
                diff_elec_vs_top5 = top5_data - electrode_data
                stat3, pval3 = wilcoxon(diff_elec_vs_top5, alternative='greater')
                
                # Store for FDR correction
                all_pvalues.extend([pval1, pval2, pval3])
                all_stats.extend([stat1, stat2, stat3])
                test_labels.extend([
                    f"{electrode}_All_vs_Top5",
                    f"{electrode}_Elec_vs_All", 
                    f"{electrode}_Elec_vs_Top5"
                ])
                
                # Store electrode results
                electrode_results[electrode] = {
                    'electrode_data': electrode_data,
                    'all_sources_data': all_sources_data,
                    'top5_data': top5_data,
                    'stats': [stat1, stat2, stat3],
                    'pvals_raw': [pval1, pval2, pval3]
                }
    
    # Apply FDR correction (Benjamini-Hochberg)
    if len(all_pvalues) > 0:
        fdr_rejected, pvals_corrected, alpha_sidak, alpha_bonf = multipletests(
            all_pvalues, alpha=0.05, method='fdr_bh', is_sorted=False
        )
        
        # Print results with FDR correction
        pval_idx = 0
        for electrode in target_electrodes:
            if electrode in electrode_results:
                result = electrode_results[electrode]
                
                # Get corrected p-values for this electrode
                pval1_corr = pvals_corrected[pval_idx]
                pval2_corr = pvals_corrected[pval_idx + 1]
                pval3_corr = pvals_corrected[pval_idx + 2]
                
                # Get significance after FDR correction
                sig1_fdr = fdr_rejected[pval_idx]
                sig2_fdr = fdr_rejected[pval_idx + 1]
                sig3_fdr = fdr_rejected[pval_idx + 2]
                
                pval_idx += 3
                
                electrode_data = result['electrode_data']
                all_sources_data = result['all_sources_data']
                top5_data = result['top5_data']
                stat1, stat2, stat3 = result['stats']
                pval1, pval2, pval3 = result['pvals_raw']
                
                print(f"\n{electrode} Results:")
                print(f"  Electrode:    {np.mean(electrode_data):.3f} ± {np.std(electrode_data):.3f}")
                print(f"  All Sources:  {np.mean(all_sources_data):.3f} ± {np.std(all_sources_data):.3f}")
                print(f"  Top 10%:       {np.mean(top5_data):.3f} ± {np.std(top5_data):.3f}")
                print(f"  ")
                print(f"  Wilcoxon Tests (Raw p-values):")
                print(f"    All vs Top 10%:    W = {stat1:.3f}, p = {pval1:.6f}")
                print(f"    Electrode vs All: W = {stat2:.3f}, p = {pval2:.6f}")
                print(f"    Electrode vs Top 10%: W = {stat3:.3f}, p = {pval3:.6f}")
                print(f"  ")
                print(f"  FDR-Corrected p-values:")
                print(f"    All vs Top 10%:    p_FDR = {pval1_corr:.6f} {'*' if sig1_fdr else ''}")
                print(f"    Electrode vs All: p_FDR = {pval2_corr:.6f} {'*' if sig2_fdr else ''}")
                print(f"    Electrode vs Top 10%: p_FDR = {pval3_corr:.6f} {'*' if sig3_fdr else ''}")
                
                # Effect sizes (mean differences)
                mean_diff_all_top5 = np.mean(top5_data) - np.mean(all_sources_data)
                mean_diff_elec_all = np.mean(all_sources_data) - np.mean(electrode_data)
                mean_diff_elec_top5 = np.mean(top5_data) - np.mean(electrode_data)
                
                print(f"  Mean Differences:")
                print(f"    Top 10% - All:     {mean_diff_all_top5:.3f}")
                print(f"    All - Electrode:  {mean_diff_elec_all:.3f}")
                print(f"    Top 10% - Electrode: {mean_diff_elec_top5:.3f}")
        
        # Summary table with FDR correction
        print(f"\n{'='*100}")
        print("FDR CORRECTION SUMMARY")
        print(f"{'='*100}")
        print(f"Total tests performed: {len(all_pvalues)}")
        print(f"Tests significant before FDR: {np.sum(np.array(all_pvalues) < 0.05)}")
        print(f"Tests significant after FDR:  {np.sum(fdr_rejected)}")
        print(f"FDR method: Benjamini-Hochberg")
        print(f"Alpha level: 0.05")
        
        print(f"\nDetailed FDR Results:")
        print(f"{'Test':<25} {'Raw p':<12} {'FDR p':<12} {'Significant'}")
        print(f"{'-'*60}")
        for i, (label, raw_p, fdr_p, sig) in enumerate(zip(test_labels, all_pvalues, pvals_corrected, fdr_rejected)):
            print(f"{label:<25} {raw_p:<12.6f} {fdr_p:<12.6f} {'Yes' if sig else 'No'}")

    print(f"\n* p_FDR < 0.05 (FDR-corrected Wilcoxon signed-rank test)")
    print(f"Positive differences indicate higher SNR in the first group.")
    print(f"FDR correction accounts for {len(all_pvalues)} multiple comparisons.")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

#-----------------------------------------------------------------------------------------#
def analyze_electrode_s1_source_snr(p):
    """
    Analyze SNR enhancement from electrode to source level in S1 regions.
    Shows which S1 regions demonstrate SNR gain when moving from scalp to source space.
    
    Focus: Demonstrate that source localization provides SNR benefits for specific S1 regions.
    
    Parameters:
    -----------
    p : parameter object
        Contains all the path and analysis parameters
    
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame with region-level SNR comparisons
    enhancement_summary : dict
        Summary of SNR enhancement by region
    """
    print("="*70)
    print("S1 SNR ENHANCEMENT ANALYSIS: Electrode → Source Space")
    print("="*70)
    
    cwd = getcwd()
    subjects = p.subjects
    rep = 2000
    method = 'eLORETA'
    dist = 5
    
    # Define S1 core somatosensory regions (primary areas)
    s1_core_regions = {
        'L_3b': ['L_3b_ROI-lh'],  # Primary somatosensory cortex
        'R_3b': ['R_3b_ROI-rh'],
        'L_1': ['L_1_ROI-lh'],    # Primary somatosensory cortex
        'R_1': ['R_1_ROI-rh'],
        'L_2': ['L_2_ROI-lh'],    # Primary somatosensory cortex
        'R_2': ['R_2_ROI-rh'],
        'L_4': ['L_4_ROI-lh'],    # Motor/sensory
        'R_4': ['R_4_ROI-rh']
    }
    
    # Electrodes overlying sensorimotor regions
    target_electrodes = ['CP3', 'CPz', 'CP4', 'C3', 'Cz', 'C4']
    
    all_data = []
    
    for subj in subjects:
        print(f"\nProcessing {subj}...")
        
        subj_number = ''.join(filter(str.isdigit, subj))
        
        # File paths
        snr_file = f"snr_{subj_number}_el.csv"
        snr_path = path.join(cwd, p.data, p.processed_data, p.protocol, subj, p.session, p.current_dir, snr_file)
        erp_path = path.join(cwd, p.data, p.processed_data, p.protocol, subj, p.session, p.eeg_dir, p.epochsfile)
        stc_file = f"current{rep}reps_{dist}mm_{method}-lh.stc"
        stc_path = path.join(cwd, p.data, p.processed_data, p.protocol, subj, p.session, p.current_dir, stc_file)
        
        if not all(path.exists(f) for f in [snr_path, erp_path, stc_path]):
            print(f"  Missing files, skipping...")
            continue
        
        try:
            # Load data
            electrode_df = pd.read_csv(snr_path)
            electrode_data = electrode_df[electrode_df["n_select"] == rep]
            stc = mne.read_source_estimate(stc_path)
            source_snr = snr_hammerer(p, stc.data, stc.times)
            
            # Get S1 region masks
            region_data = {}
            for region_name, patterns in s1_core_regions.items():
                mask = get_s1_source_mask(p, patterns, stc)
                if mask is not None and np.any(mask):
                    region_snr = source_snr[mask]
                    region_data[region_name] = {
                        'mean': np.mean(region_snr),
                        'median': np.median(region_snr),
                        'std': np.std(region_snr),
                        'n_sources': np.sum(mask)
                    }
            
            if not region_data:
                print(f"  No S1 regions found, skipping...")
                continue
            
            # Compare each electrode to relevant regions
            for electrode in target_electrodes:
                elec_row = electrode_data[electrode_data["channel"] == electrode]
                if elec_row.empty:
                    continue
                
                electrode_snr = elec_row["snr_mean"].iloc[0]
                
                # Determine relevant hemisphere
                if electrode in ['CP3', 'C3']:
                    relevant_regions = [r for r in region_data.keys() if r.startswith('L_')]
                elif electrode in ['CP4', 'C4']:
                    relevant_regions = [r for r in region_data.keys() if r.startswith('R_')]
                else:  # CPz, Cz - compare to both
                    relevant_regions = list(region_data.keys())
                
                # Store comparisons
                for region in relevant_regions:
                    if region in region_data:
                        source_mean = region_data[region]['mean']
                        enhancement = ((source_mean - electrode_snr) / electrode_snr) * 100
                        
                        all_data.append({
                            'subject': subj,
                            'electrode': electrode,
                            'region': region,
                            'hemisphere': 'Left' if region.startswith('L_') else 'Right',
                            'electrode_snr': electrode_snr,
                            'source_snr_mean': source_mean,
                            'source_snr_median': region_data[region]['median'],
                            'snr_ratio': source_mean / electrode_snr,
                            'enhancement_pct': enhancement,
                            'n_sources': region_data[region]['n_sources']
                        })
        
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    results_df = pd.DataFrame(all_data)
    
    if results_df.empty:
        print("\nNo data collected!")
        return results_df, {}
    
    print(f"\n Collected {len(results_df)} comparisons from {results_df['subject'].nunique()} subjects")
    
    # Compute enhancement summary
    enhancement_summary = compute_enhancement_summary(results_df)
    
    # Create focused visualizations
    create_enhancement_visualizations(results_df, enhancement_summary)
    
    return results_df, enhancement_summary


def compute_enhancement_summary(results_df):
    """
    Compute statistical summary of SNR enhancement by region.
    """
    from scipy import stats
    
    print("\n" + "="*70)
    print("SNR ENHANCEMENT SUMMARY")
    print("="*70)
    
    summary = {}
    
    # Group by region
    for region in results_df['region'].unique():
        region_data = results_df[results_df['region'] == region]
        
        electrode_snrs = region_data['electrode_snr'].values
        source_snrs = region_data['source_snr_mean'].values
        enhancements = region_data['enhancement_pct'].values
        
        # One-sample t-test: is enhancement significantly different from 0?
        t_stat, p_val = stats.ttest_1samp(enhancements, 0)
        
        # Compute effect size
        cohens_d = np.mean(enhancements) / np.std(enhancements) if np.std(enhancements) > 0 else 0
        
        # Is enhancement positive and significant?
        is_enhanced = np.mean(enhancements) > 0 and p_val < 0.05
        
        summary[region] = {
            'mean_electrode_snr': np.mean(electrode_snrs),
            'mean_source_snr': np.mean(source_snrs),
            'mean_ratio': np.mean(source_snrs) / np.mean(electrode_snrs),
            'mean_enhancement_pct': np.mean(enhancements),
            'median_enhancement_pct': np.median(enhancements),
            'std_enhancement': np.std(enhancements),
            'pct_enhanced': np.sum(enhancements > 0) / len(enhancements) * 100,
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'is_enhanced': is_enhanced,
            'n_comparisons': len(region_data)
        }
    
    # Print summary table
    print(f"\n{'Region':<8} {'Electrode':<10} {'Source':<10} {'Ratio':<7} {'Enhancement':<12} {'% Improved':<12} {'p-value':<10} {'Significant'}")
    print("-"*90)
    
    # Sort by enhancement
    sorted_regions = sorted(summary.items(), key=lambda x: x[1]['mean_enhancement_pct'], reverse=True)
    
    for region, stats in sorted_regions:
        sig_marker = "***" if stats['is_enhanced'] else ""
        print(f"{region:<8} {stats['mean_electrode_snr']:<10.2f} {stats['mean_source_snr']:<10.2f} "
              f"{stats['mean_ratio']:<7.2f} {stats['mean_enhancement_pct']:>+6.1f}% "
              f"{stats['pct_enhanced']:>10.1f}% {stats['p_value']:<10.4f} {sig_marker}")
    
    print("\n*** = Significant positive enhancement (p < 0.05)")
    print("Enhancement = ((Source SNR - Electrode SNR) / Electrode SNR) × 100%")
    print("% Improved = Percentage of subjects showing positive enhancement")
    
    # Key findings
    enhanced_regions = [r for r, s in summary.items() if s['is_enhanced']]
    if enhanced_regions:
        print(f"\n📈 REGIONS WITH SIGNIFICANT SNR ENHANCEMENT ({len(enhanced_regions)}):")
        for region in enhanced_regions:
            s = summary[region]
            print(f"   • {region}: +{s['mean_enhancement_pct']:.1f}% (p={s['p_value']:.4f}, d={s['cohens_d']:.2f})")
    else:
        print("\n⚠ No regions showed significant SNR enhancement")
    
    return summary


def create_enhancement_visualizations(results_df, enhancement_summary):
    """
    Create focused visualizations showing SNR enhancement.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    
    # Color palette
    colors_hemi = {'Left': '#3498db', 'Right': '#e74c3c'}
    
    # Sort regions by enhancement
    sorted_regions = sorted(enhancement_summary.items(), 
                          key=lambda x: x[1]['mean_enhancement_pct'], 
                          reverse=True)
    region_order = [r[0] for r in sorted_regions]
    
    # ========================================================================
    # FIGURE 1: Enhancement Percentage by Region (Main Result)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    enhancements = [enhancement_summary[r]['mean_enhancement_pct'] for r in region_order]
    p_values = [enhancement_summary[r]['p_value'] for r in region_order]
    hemisphere_colors = [colors_hemi[results_df[results_df['region']==r]['hemisphere'].iloc[0]] 
                        for r in region_order]
    
    bars = ax.barh(range(len(region_order)), enhancements, 
                   color=hemisphere_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add significance stars
    for i, (enh, p_val) in enumerate(zip(enhancements, p_values)):
        if p_val < 0.001:
            marker = '***'
        elif p_val < 0.01:
            marker = '**'
        elif p_val < 0.05:
            marker = '*'
        else:
            marker = ''
        
        if marker:
            x_pos = enh + (2 if enh > 0 else -2)
            ax.text(x_pos, i, marker, ha='left' if enh > 0 else 'right', 
                   va='center', fontsize=14, fontweight='bold')
    
    # Add zero line
    ax.axvline(0, color='black', linestyle='-', linewidth=2, zorder=0)
    
    # Styling
    ax.set_yticks(range(len(region_order)))
    ax.set_yticklabels(region_order, fontsize=12)
    ax.set_xlabel('SNR Enhancement (%)', fontsize=14, fontweight='bold')
    ax.set_title('Source Localization SNR Enhancement by S1 Region\n(Compared to Scalp Electrodes)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_hemi['Left'], edgecolor='black', label='Left Hemisphere'),
        Patch(facecolor=colors_hemi['Right'], edgecolor='black', label='Right Hemisphere')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # FIGURE 2: Before/After Comparison
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Prepare data for before/after comparison
    plot_data = []
    for region in region_order:
        region_data = results_df[results_df['region'] == region]
        
        for _, row in region_data.iterrows():
            plot_data.append({
                'Region': region,
                'Space': 'Electrode',
                'SNR': row['electrode_snr'],
                'Hemisphere': row['hemisphere']
            })
            plot_data.append({
                'Region': region,
                'Space': 'Source',
                'SNR': row['source_snr_mean'],
                'Hemisphere': row['hemisphere']
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Left panel: Box plot comparison
    ax = axes[0]
    sns.boxplot(data=plot_df, x='Region', y='SNR', hue='Space', 
               order=region_order, palette=['lightgray', 'coral'], ax=ax)
    ax.set_xticklabels(region_order, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('SNR', fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    ax.set_title('Electrode vs Source SNR by Region', fontsize=14, fontweight='bold')
    ax.legend(title='Measurement Space', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Right panel: Ratio plot
    ax = axes[1]
    ratios = [enhancement_summary[r]['mean_ratio'] for r in region_order]
    bars = ax.barh(range(len(region_order)), ratios, 
                   color=hemisphere_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, ratio in enumerate(ratios):
        ax.text(ratio + 0.02, i, f'{ratio:.2f}', 
               va='center', fontsize=11, fontweight='bold')
    
    ax.axvline(1, color='red', linestyle='--', linewidth=2, label='Equal SNR', zorder=0)
    ax.set_yticks(range(len(region_order)))
    ax.set_yticklabels(region_order, fontsize=12)
    ax.set_xlabel('SNR Ratio (Source / Electrode)', fontsize=13, fontweight='bold')
    ax.set_title('Source-to-Electrode SNR Ratio', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # FIGURE 3: Individual Subject Enhancement Patterns
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get top 4 regions with highest enhancement
    top_regions = region_order[:4]
    
    x_offset = 0
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_regions)))
    
    for idx, region in enumerate(top_regions):
        region_data = results_df[results_df['region'] == region]
        enhancements = region_data['enhancement_pct'].values
        
        # Create violin plot
        parts = ax.violinplot([enhancements], positions=[x_offset], widths=0.7,
                             showmeans=True, showmedians=True)
        
        # Color the violin
        for pc in parts['bodies']:
            pc.set_facecolor(colors[idx])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Add individual points
        jitter = np.random.normal(0, 0.04, len(enhancements))
        ax.scatter(x_offset + jitter, enhancements, alpha=0.4, s=30, 
                  color=colors[idx], edgecolors='black', linewidth=0.5, zorder=3)
        
        # Add mean line
        mean_val = np.mean(enhancements)
        ax.hlines(mean_val, x_offset-0.35, x_offset+0.35, 
                 colors='red', linestyles='-', linewidth=3, zorder=4)
        
        x_offset += 1
    
    # Add zero line
    ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
    
    # Styling
    ax.set_xticks(range(len(top_regions)))
    ax.set_xticklabels(top_regions, fontsize=13, fontweight='bold')
    ax.set_ylabel('SNR Enhancement (%)', fontsize=14, fontweight='bold')
    ax.set_title('Subject-Level Enhancement Distribution (Top 4 Regions)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # Add text box with interpretation
    textstr = 'Points above 0: Source SNR > Electrode SNR\nPoints below 0: Electrode SNR > Source SNR'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # FIGURE 4: Summary Statistics Heatmap
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare matrix
    metrics = ['mean_ratio', 'mean_enhancement_pct', 'pct_enhanced', 'cohens_d']
    metric_labels = ['SNR Ratio', 'Enhancement (%)', '% Improved', "Cohen's d"]
    
    matrix = np.zeros((len(region_order), len(metrics)))
    for i, region in enumerate(region_order):
        for j, metric in enumerate(metrics):
            matrix[i, j] = enhancement_summary[region][metric]
    
    # Normalize each column for visualization
    matrix_norm = matrix.copy()
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        matrix_norm[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-10)
    
    sns.heatmap(matrix_norm, annot=matrix, fmt='.2f',
               xticklabels=metric_labels, yticklabels=region_order,
               cmap='RdYlGn', center=0.5, cbar_kws={'label': 'Normalized Score'},
               linewidths=1, linecolor='gray', ax=ax)
    
    ax.set_title('SNR Enhancement Metrics by Region', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------------------------------------#
#---------------------------  FIGURE 3 CODE -----------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree, distance_matrix
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Rectangle
from mne.label import Label


#------------------------------------------------------------------------------------#
def make_hand_disk_label_both_hemis(subject, subjects_dir, parc='HCPMMP1', radius_mm=8.0):
    """
    Build hand disk labels for BOTH hemispheres.
    LH: right hand area (3b,1,2)
    RH: left hand area (mirrored, 3b,1,2)
    
    Returns: (lh_label, lh_center_vert, rh_label, rh_center_vert)
    """
    print(f"\n[make_hand_disk_label_both_hemis] Building hand disks | radius={radius_mm} mm")
    
    try:
        labels = mne.read_labels_from_annot(subject=subject, parc=parc, 
                                           hemi='both', subjects_dir=subjects_dir)
    except Exception as e:
        print(f" Could not read annotation '{parc}': {e}")
        return None, None, None, None
    
    results = {}
    
    # Process both hemispheres
    for hemi in ['lh', 'rh']:
        hemi_prefix = 'L_' if hemi == 'lh' else 'R_'
        s1_patterns = [f'{hemi_prefix}3b_ROI-{hemi}', 
                      f'{hemi_prefix}1_ROI-{hemi}', 
                      f'{hemi_prefix}2_ROI-{hemi}']
        
        hemi_labels = [lbl for lbl in labels if lbl.name in s1_patterns]
        
        if not hemi_labels:
            hemi_labels = [lbl for lbl in labels if lbl.name.startswith(hemi_prefix) and
                          any(x in lbl.name for x in ['3b', '_1_', '_2_'])]
        
        if not hemi_labels:
            print(f"  No {hemi.upper()} S1 labels found")
            results[hemi] = (None, None)
            continue
        
        # Combine vertices
        hemi_vertices = np.unique(np.hstack([lbl.vertices for lbl in hemi_labels]))
        
        # Load surface
        surf_fname = path.join(subjects_dir, subject, 'surf', f'{hemi}.white')
        try:
            coords, faces = mne.read_surface(surf_fname)
        except Exception as e:
            print(f" Could not read surface '{surf_fname}': {e}")
            results[hemi] = (None, None)
            continue
        
        # Compute centroid and disk on white surface
        centroid = coords[hemi_vertices].mean(axis=0)
        dists = np.linalg.norm(coords - centroid, axis=1)
        selected_vertices = np.where(dists <= radius_mm)[0]
        
        if selected_vertices.size == 0:
            print(f" No {hemi.upper()} vertices within {radius_mm} mm")
            results[hemi] = (None, None)
            continue
        
        # Load inflated surface
        inflated_fname = path.join(subjects_dir, subject, 'surf', f'{hemi}.inflated')
        try:
            coords_inflated, _ = mne.read_surface(inflated_fname)
        except Exception as e:
            print(f" Could not read inflated surface '{inflated_fname}': {e}")
            results[hemi] = (None, None)
            continue
        
        # Build label using inflated coordinates
        try:
            disk_label = Label(selected_vertices, pos=coords_inflated[selected_vertices],
                               hemi=hemi, name=f'S1_hand_disk_{hemi}', subject=subject)
        except Exception:
            disk_label = Label(selected_vertices, pos=coords_inflated[selected_vertices])
            disk_label.hemi = hemi
            disk_label.name = f'S1_hand_disk_{hemi}'
            disk_label.subject = subject
        
        # Center vertex remains based on white surface distances
        center_vertex = int(np.argmin(dists))

        
        print(f" {hemi.upper()} hand disk: {len(selected_vertices)} vertices | center={center_vertex}")
        results[hemi] = (disk_label, center_vertex)
    
    lh_label, lh_center = results.get('lh', (None, None))
    rh_label, rh_center = results.get('rh', (None, None))
    
    return lh_label, lh_center, rh_label, rh_center

#------------------------------------------------------------------------------------#
def compute_hemisphere_geodesics(p, hemi, center_vert, stc_data, stc_vertno, stc_times):
    """
    Compute geodesic distances for one hemisphere.
    
    Parameters
    ----------
    p : params object
    hemi : str ('lh' or 'rh')
    center_vert : int
        Center vertex index for the hand area
    stc_data : array
        SNR data for this hemisphere (vertices × time or 1D)
    stc_vertno : array
        Source vertex indices for this hemisphere
    stc_times : array
        Time points for the STC
        
    Returns
    -------
    source_geo_dists : array
        Geodesic distances for each source vertex (1D)
    snr_data : array
        SNR values (1D)
    """
    print(f"\n[compute_hemisphere_geodesics] Processing {hemi.upper()}...")
    
    # Load cortical surface
    surf_path = path.join(p.subj_brains_dir, p.fs_subject_dir, "surf", f"{hemi}.pial")
    coords, faces = mne.read_surface(surf_path)
    print(f" {hemi.upper()} pial surface: {len(coords)} vertices")

    # Build adjacency graph
    print(f" Building {hemi.upper()} cortical adjacency graph...")
    graph = build_cortical_graph(coords, faces)

    # Compute geodesic distances
    if center_vert is not None:
        print(f" Computing geodesic distances from vertex {center_vert}...")
        geo_dists_full = dijkstra(csgraph=graph, directed=False, indices=center_vert)
        valid = np.isfinite(geo_dists_full)
        print(f" Geodesic range: [{np.nanmin(geo_dists_full[valid]):.1f}, {np.nanmax(geo_dists_full[valid]):.1f}] mm")
        print(f" Valid: {np.sum(valid)}/{len(geo_dists_full)}")
    else:
        print(f" WARNING: No center vertex for {hemi.upper()}")
        geo_dists_full = np.full(len(coords), np.nan)

    # Map distances to source vertices
    print(f" Source space: {len(stc_vertno)} {hemi.upper()} vertices")
    if stc_vertno.max() >= len(coords):
        print(f" ERROR: Source indices exceed surface size!")
        source_geo_dists = np.full(len(stc_vertno), np.nan)
    else:
        source_geo_dists = geo_dists_full[stc_vertno]

    # ------------------------------------------------------------------
    # FIX: Treat incoming data as already computed SNR
    # ------------------------------------------------------------------
    print(f" Using provided SNR data (no recomputation): {stc_data.shape}")

    # If 2D (vertices × time), average over time
    if stc_data.ndim > 1:
        snr_data = np.nanmean(stc_data, axis=1)
    else:
        snr_data = stc_data

    # Flatten to ensure 1D arrays
    snr_data = snr_data.flatten()
    source_geo_dists = source_geo_dists.flatten()

    print(f" Final SNR stats:")
    print(f"   SNR shape: {snr_data.shape}")
    print(f"   SNR range: [{np.nanmin(snr_data):.6f}, {np.nanmax(snr_data):.6f}]")
    print(f"   Valid SNR: {np.sum(np.isfinite(snr_data))}/{len(snr_data)}")
    print(f" Final: {len(source_geo_dists)} distances, {len(snr_data)} SNR values")

    return source_geo_dists, snr_data

#------------------------------------------------------------------------------------#
def collect_hemisphere_electrode_data(subject_data, subjects, rep, hemi, 
                                      geo_dists_scalp, electrode_positions, 
                                      electrode_names, min_subjects=1):  # Changed from implicit filtering
    """
    Collect electrode data for one hemisphere (including midline).
    
    Parameters
    ----------
    min_subjects : int
        Minimum number of subjects required for an electrode to be included (default: 1)
    """
    print(f"\n[collect_hemisphere_electrode_data] Processing {hemi.upper()} electrodes...")
    
    def is_electrode_in_hemisphere(elec_name, target_hemi):
        """Check if electrode belongs to target hemisphere."""
        elec_upper = elec_name.upper()
        
        # Midline electrodes (always include)
        if elec_upper.endswith('Z'):
            return True
        
        # Left hemisphere: odd numbers
        if target_hemi == 'lh':
            return any(elec_upper.endswith(str(i)) for i in [1, 3, 5, 7, 9])
        
        # Right hemisphere: even numbers
        elif target_hemi == 'rh':
            return any(elec_upper.endswith(str(i)) for i in [2, 4, 6, 8, 0])
        
        return False
    
    # Collect all available electrodes across subjects
    all_available_electrodes = set()
    for subj in subjects:
        if subj not in subject_data:
            continue
        df_subj = subject_data[subj].get('electrode_snr', None)
        if df_subj is None:
            continue
        df_rep = df_subj[df_subj["n_select"] == rep]
        if df_rep.empty:
            continue
        available = set(df_rep["channel"].unique())
        all_available_electrodes.update(available)
    
    # Filter by hemisphere
    electrodes_to_process = {e for e in all_available_electrodes 
                            if is_electrode_in_hemisphere(e, hemi)}
    
    print(f" Found {len(electrodes_to_process)} {hemi.upper()} electrodes in data")
    
    # Collect data
    electrode_data = {elec: {'snr': [], 'dist': []} for elec in electrodes_to_process}
    electrode_positions_dict = {}
    
    montage = mne.channels.make_standard_montage("standard_1005")
    
    for subj in subjects:
        if subj not in subject_data:
            continue
        df_subj = subject_data[subj].get('electrode_snr', None)
        if df_subj is None:
            continue
        df_rep = df_subj[df_subj["n_select"] == rep]
        if df_rep.empty:
            continue
        
        subj_info = subject_data[subj]['erp_info'].copy()
        subj_info.set_montage(montage, match_case=False, on_missing='ignore')
        
        for electrode in electrodes_to_process:
            if electrode not in subj_info['ch_names']:
                continue
            
            try:
                ch_idx = subj_info['ch_names'].index(electrode)
                electrode_pos = np.array(subj_info['chs'][ch_idx]['loc'][:3])
                if np.allclose(electrode_pos, 0):
                    continue
                
                electrode_pos_mm = electrode_pos * 1000
                
                if electrode not in electrode_positions_dict:
                    electrode_positions_dict[electrode] = []
                electrode_positions_dict[electrode].append(electrode_pos_mm)
                
                snr_val = df_rep.loc[df_rep["channel"] == electrode, "snr_mean"]
                if not snr_val.empty:
                    electrode_data[electrode]['snr'].append(snr_val.values[0])
            except (ValueError, IndexError):
                continue
    
    # Compute geodesic distances using average position
    if geo_dists_scalp is not None and electrode_positions is not None:
        for elec in electrodes_to_process:
            if elec not in electrode_positions_dict or not electrode_positions_dict[elec]:
                continue
            
            # Find this electrode in the main electrode list
            try:
                elec_idx = electrode_names.index(elec)
                geo_dist = geo_dists_scalp[elec_idx]
                
                # Assign distance to all SNR measurements for this electrode
                for _ in range(len(electrode_data[elec]['snr'])):
                    electrode_data[elec]['dist'].append(geo_dist)
            except (ValueError, IndexError):
                continue
    
    # Average across subjects - include electrodes with at least min_subjects
    electrode_avg = {}
    for elec in electrodes_to_process:
        if len(electrode_data[elec]['snr']) >= min_subjects and electrode_data[elec]['dist']:
            electrode_avg[elec] = {
                'snr': np.nanmean(electrode_data[elec]['snr']),
                'dist': np.nanmean(electrode_data[elec]['dist']),
                'snr_std': np.nanstd(electrode_data[elec]['snr']),
                'n': len(electrode_data[elec]['snr'])
            }
    
    print(f" Total {hemi.upper()} electrodes with data (≥{min_subjects} subjects): {len(electrode_avg)}")
    
    return electrode_avg

#------------------------------------------------------------------------------------#

    
#------------------------------------------------------------------------------------#
def compute_significance_markers(bin_centers, subject_bin_data, alpha=0.05, verbose=True):
    """
    Compute statistical significance using Wilcoxon signed-rank test with FDR correction.
    Test against the first bin (closest to hand area).
    
    Parameters
    ----------
    bin_centers : array
        Center positions of bins
    subject_bin_data : dict
        {bin_center: [subject1_mean, subject2_mean, ...]}
    alpha : float
        FDR significance level
    verbose : bool
        Print detailed statistics
        
    Returns
    -------
    significant_bins : array of bool
        True for bins significantly different from first bin
    """
    print(f"\n[compute_significance_markers] Testing significance with FDR α={alpha}")
    
    # Get baseline (first bin) data across subjects
    baseline_data = np.array(subject_bin_data[bin_centers[0]])
    baseline_data = baseline_data[np.isfinite(baseline_data)]
    
    if len(baseline_data) < 3:
        print(" WARNING: Not enough subjects for statistics (need ≥3)")
        return np.zeros(len(bin_centers), dtype=bool)
    
    print(f" Baseline bin (0-{bin_centers[0]*2:.0f}mm): N={len(baseline_data)}, "
          f"mean={np.mean(baseline_data):.3f}, std={np.std(baseline_data):.3f}")
    
    # Test each bin against baseline
    p_values = []
    n_subjects_per_bin = []
    mean_diffs = []
    
    for i, bc in enumerate(bin_centers):
        bin_data = np.array(subject_bin_data[bc])
        bin_data = bin_data[np.isfinite(bin_data)]
        n_subjects_per_bin.append(len(bin_data))
        
        if len(bin_data) < 3:
            p_values.append(1.0)
            mean_diffs.append(np.nan)
            continue
        
        # Check if we have paired data (same subjects in both bins)
        if len(bin_data) != len(baseline_data):
            if verbose and i < 5:  # Only print for first few bins
                print(f" WARNING: Bin {i} has {len(bin_data)} subjects vs baseline {len(baseline_data)}")
            p_values.append(1.0)
            mean_diffs.append(np.nan)
            continue
        
        # Compute mean difference
        diff = np.mean(baseline_data) - np.mean(bin_data)
        mean_diffs.append(diff)
        
        # Wilcoxon signed-rank test (paired)
        try:
            # Check if there's any difference
            if np.allclose(baseline_data, bin_data):
                p_values.append(1.0)
            else:
                statistic, p_val = wilcoxon(baseline_data, bin_data, alternative='two-sided')
                p_values.append(p_val)
                
                if verbose and i < 5:  # Print first few bins
                    print(f" Bin {i} ({bc:.0f}mm): mean_diff={diff:.3f}, "
                          f"W={statistic:.1f}, p={p_val:.4f}")
        except Exception as e:
            if verbose and i < 5:
                print(f" Bin {i}: Wilcoxon failed: {e}")
            p_values.append(1.0)
    
    p_values = np.array(p_values)
    
    # FDR correction
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    # Summary statistics
    n_sig = np.sum(reject)
    n_tested = np.sum(~np.isnan(mean_diffs))
    
    print(f"\n Statistical Summary:")
    print(f"  Total bins: {len(bin_centers)}")
    print(f"  Bins with sufficient data: {n_tested}")
    print(f"  Significant bins (FDR q<{alpha}): {n_sig}")
    
    if n_sig > 0:
        sig_indices = np.where(reject)[0]
        print(f"  Significant bin distances: {bin_centers[sig_indices]}")
        print(f"  Mean differences at significant bins: {np.array(mean_diffs)[sig_indices]}")
        print(f"  Uncorrected p-values: {p_values[sig_indices]}")
        print(f"  Corrected p-values: {pvals_corrected[sig_indices]}")
    
    # Additional validation
    if verbose:
        print(f"\n Validation checks:")
        print(f"  All p-values in [0,1]: {np.all((p_values >= 0) & (p_values <= 1))}")
        print(f"  Number of p=1.0 (insufficient data): {np.sum(p_values == 1.0)}")
        
        # Check if baseline decreasing trend exists
        valid_means = [np.nanmean(subject_bin_data[bc]) for bc in bin_centers[:10]]
        valid_means = [m for m in valid_means if not np.isnan(m)]
        if len(valid_means) > 5:
            # Simple monotonicity check
            decreasing = np.sum(np.diff(valid_means) < 0)
            print(f"  First 10 bins show decreasing trend: {decreasing}/{len(valid_means)-1} bins")
    
    return reject

#------------------------------------------------------------------------------------#
def compute_electrode_moving_average(electrode_avg, window_size=3):
    """
    Compute moving average of electrode SNR values.
    
    Returns
    -------
    sorted_dists : array
    moving_avg : array
    """
    if not electrode_avg:
        return np.array([]), np.array([])
    
    dists = np.array([v['dist'] for v in electrode_avg.values()])
    snrs = np.array([v['snr'] for v in electrode_avg.values()])
    
    # Sort by distance
    sort_idx = np.argsort(dists)
    sorted_dists = dists[sort_idx]
    sorted_snrs = snrs[sort_idx]
    
    # Compute moving average
    moving_avg = np.convolve(sorted_snrs, np.ones(window_size)/window_size, mode='same')
    
    # Fix edges
    for i in range(min(window_size//2, len(moving_avg))):
        moving_avg[i] = np.mean(sorted_snrs[:i+window_size//2+1])
        moving_avg[-(i+1)] = np.mean(sorted_snrs[-(i+window_size//2+1):])
    
    return sorted_dists, moving_avg

#---------------------------------------------------------------------------------------#
def compute_binned_statistics(all_subject_source_data, bin_size=20.0, max_dist=500, min_points_per_bin=8):
    """
    Bin source data and store ALL individual points (not averaged per subject).
    
    Parameters:
    -----------
    all_subject_source_data : list
        List of (geo_dists, snr_values) tuples for each subject
    bin_size : float
        Width of each bin in mm
    max_dist : float
        Maximum distance to consider
    min_points_per_bin : int
        Minimum number of data points required per bin to compute statistics
    
    Returns:
    --------
    bin_centers : array
        Center position of each bin
    bin_means : array
        Mean SNR per bin (NaN if insufficient data)
    bin_stds : array
        Standard deviation per bin (NaN if insufficient data)
    bin_raw_data : dict
        ALL individual source points in each bin (not per-subject averages)
    """
    bins = np.arange(0, max_dist + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Store ALL individual points per bin (not averaged per subject)
    bin_raw_data = {bc: [] for bc in bin_centers}
    
    print(f"\n[compute_binned_statistics] Binning with bin_size={bin_size}mm, max_dist={max_dist}mm")
    print(f"  Minimum points per bin required: {min_points_per_bin}")
    
    total_points = 0
    for subj_idx, (geo_dists, snr_values) in enumerate(all_subject_source_data):
        geo_dists = np.asarray(geo_dists).flatten()
        snr_values = np.asarray(snr_values).flatten()
        
        # Ensure same length
        if geo_dists.size != snr_values.size:
            min_size = min(geo_dists.size, snr_values.size)
            geo_dists = geo_dists[:min_size]
            snr_values = snr_values[:min_size]
        
        # Filter valid data
        valid = np.isfinite(geo_dists) & np.isfinite(snr_values)
        geo_dists_valid = geo_dists[valid]
        snr_valid = snr_values[valid]
        
        total_points += len(geo_dists_valid)
        
        if len(geo_dists_valid) == 0:
            print(f"  Subject {subj_idx+1}: No valid data!")
            continue
        
        # Bin the data - store ALL individual points
        for bc, bin_start, bin_end in zip(bin_centers, bins[:-1], bins[1:]):
            in_bin = (geo_dists_valid >= bin_start) & (geo_dists_valid < bin_end)
            if np.any(in_bin):
                # Store all individual SNR values in this bin
                bin_raw_data[bc].extend(snr_valid[in_bin].tolist())
    
    print(f"  Total valid source data points: {total_points}")
    
    # Compute statistics per bin
    bin_means = []
    bin_stds = []
    bin_counts = []
    bins_below_threshold = 0
    
    for bc in bin_centers:
        n_points = len(bin_raw_data[bc])
        
        if n_points >= min_points_per_bin:
            bin_means.append(np.mean(bin_raw_data[bc]))
            bin_stds.append(np.std(bin_raw_data[bc]))
            bin_counts.append(n_points)
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            bin_counts.append(0)
            if n_points > 0:
                bins_below_threshold += 1
    
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    bin_counts = np.array(bin_counts)
    
    # Report statistics
    valid_bins = np.sum(~np.isnan(bin_means))
    print(f"  Binned statistics: {valid_bins}/{len(bin_centers)} bins have sufficient data")
    print(f"  Bins below threshold: {bins_below_threshold}")
    if np.any(bin_counts > 0):
        print(f"  Points per bin: min={np.min(bin_counts[bin_counts > 0]):.0f}, "
              f"max={np.max(bin_counts):.0f}, mean={np.mean(bin_counts[bin_counts > 0]):.1f}")
    
    return bin_centers, bin_means, bin_stds, bin_raw_data


def compute_significance_source_vs_electrode(
    bin_centers, bin_raw_source_data, all_subject_electrode_data,
    alpha=0.05, test_type='two-tailed', alternative='greater', 
    min_electrode_points=5, bin_size=20.0):
    """
    Test significance using Wilcoxon signed-rank test between source and electrode SNRs within each bin.

    For each bin:
    - Collect all source SNRs
    - Collect all electrode SNRs
    - Perform Wilcoxon signed-rank test if paired data and enough points

    Parameters:
    -----------
    bin_centers : array
        Center positions of bins
    bin_raw_source_data : dict
        Dictionary mapping bin_center -> list of all source SNR values in that bin
    all_subject_electrode_data : list
        List of per-subject electrode dictionaries
    alpha : float
        Significance level for FDR correction
    test_type : str
        'two-tailed' or 'one-tailed'
    alternative : str
        For one-tailed: 'greater' (source > electrode) or 'less' (source < electrode)
    min_electrode_points : int
        Minimum number of electrode points required in a bin
    bin_size : float
        Bin width in mm
    
    Returns:
    --------
    significant_bins : array of bool
        True for bins that are significant after FDR correction
    pvals_corrected : array
        FDR-corrected p-values
    p_values : array
        Raw p-values
    """
    print(f"\n[compute_significance] Using Wilcoxon signed-rank test")
    if test_type == 'one-tailed':
        print(f"  Alternative: source SNR is {alternative} than electrode SNR")
    print(f"  Minimum electrode points per bin: {min_electrode_points}")

    p_values = []

    # First, collect all electrode data into bins
    electrode_bin_data = {bc: [] for bc in bin_centers}
    half_bin = bin_size / 2

    for subj_data in all_subject_electrode_data:
        for elec_name, elec_info in subj_data.items():
            dist = elec_info['dist']
            snr = elec_info['snr']
            for bc in bin_centers:
                if bc - half_bin <= dist < bc + half_bin:
                    electrode_bin_data[bc].append(snr)
                    break

    print(f"\n  Electrode points per bin:")
    for bc in bin_centers[:10]:
        print(f"    Bin {bc}mm: {len(electrode_bin_data[bc])} electrodes, "
              f"{len(bin_raw_source_data[bc])} sources")

    # Wilcoxon test per bin
    for bc in bin_centers:
        source_values = np.array(bin_raw_source_data[bc])
        electrode_values = np.array(electrode_bin_data[bc])

        # Check for sufficient data and pairing
        n = min(len(source_values), len(electrode_values))
        if n < min_electrode_points:
            p_values.append(np.nan)
            continue

        # Truncate to equal lengths for pairing
        source_values = source_values[:n]
        electrode_values = electrode_values[:n]

        try:
            if test_type == 'two-tailed':
                stat, p_value = wilcoxon(source_values, electrode_values, alternative='two-sided')
            elif test_type == 'one-tailed':
                stat, p_value = wilcoxon(source_values, electrode_values, alternative=alternative)
            else:
                raise ValueError(f"Unknown test_type: {test_type}")
        except ValueError as e:
            print(f"  Warning: Wilcoxon failed for bin {bc}mm: {e}")
            p_value = np.nan

        p_values.append(p_value)

    # FDR correction
    p_values = np.array(p_values, dtype=float)
    valid_mask = ~np.isnan(p_values)

    if np.any(valid_mask):
        reject, pvals_corrected_valid = fdrcorrection(p_values[valid_mask], alpha=alpha)
        pvals_corrected = np.full_like(p_values, np.nan)
        pvals_corrected[valid_mask] = pvals_corrected_valid
        significant_bins = np.full_like(p_values, False, dtype=bool)
        significant_bins[valid_mask] = reject
    else:
        pvals_corrected = np.full_like(p_values, np.nan)
        significant_bins = np.full_like(p_values, False, dtype=bool)

    sig_count = np.sum(significant_bins)
    print(f"\n  Significant bins after FDR: {sig_count}/{len(bin_centers)}")
    if sig_count > 0:
        print(f"  Significant bin centers: {bin_centers[significant_bins]}")

    return significant_bins, pvals_corrected, p_values

def plot_hemisphere_snr_distance(
    hemi, all_subject_source_data, electrode_avg, rep, method,
    n_subjects, cwd, bin_size=10.0, show_electrode_labels=True,
    label_subset=None, x_max=250, y_scale_factor=5,
    global_max_snr=None, test_type='two-tailed',
    min_points_per_bin=5
):
    """
    Create publication-quality plot for one hemisphere using seaborn styling.
    """

    # --- Debug info ---
    print(f"\n[DEBUG] Plotting {hemi.upper()} hemisphere, {rep} trials, method={method}")

    sns.set_style("white")
    sns.set_context("paper", font_scale=1.5)
    mpl.rcParams.update({
        "figure.dpi": 120,
        "axes.labelsize": 30,
        "xtick.labelsize": 28,
        "ytick.labelsize": 28,
        "legend.fontsize": 18,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })

    # --- Compute binned statistics for sources and electrodes ---
    x_cap = 200.0
    bin_centers, bin_means, bin_stds, bin_raw_data = compute_binned_statistics(
        all_subject_source_data, bin_size=bin_size, max_dist=x_cap, min_points_per_bin=min_points_per_bin
    )
    elec_bin_centers, elec_bin_means, elec_bin_stds = compute_electrode_binned_statistics(
        electrode_avg, bin_size=bin_size, max_dist=x_cap
    )

    # Convert single dict to list
    if isinstance(electrode_avg, dict) and not isinstance(electrode_avg, list):
        all_subject_electrode_data = [electrode_avg]
    else:
        all_subject_electrode_data = electrode_avg

    # Determine tail type
    if test_type == "one-tailed":
        alternative = "greater" if hemi == "lh" else "less"
    else:
        alternative = "greater"

    # --- Compute significance ---
    significant_bins, pvals_corrected, pvals_raw = compute_significance_source_vs_electrode(
        bin_centers, bin_raw_data, all_subject_electrode_data,
        alpha=0.05, test_type=test_type, alternative=alternative, bin_size=bin_size
    )

    special_names = set(label_subset) if label_subset else {"CP3", "CP4", "CPz"}
    special_data = {k: v for k, v in electrode_avg.items() if k in special_names}

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(14, 9))

    # --- Plot raw source data (gray scatter) ---
    for geo_dists, snr_values in all_subject_source_data:
        valid = np.isfinite(geo_dists) & np.isfinite(snr_values) & (geo_dists <= x_cap)
        ax.scatter(
            geo_dists[valid], snr_values[valid],
            alpha=0.9, s=10, c="lightgray", rasterized=True, zorder=1
        )

    # --- Binned source SNR ± std ---
    valid_mask = ~np.isnan(bin_means)
    if np.any(valid_mask):
        ax.plot(bin_centers[valid_mask], bin_means[valid_mask],
                color="black", linewidth=2.5, label="Source SNR", zorder=10)
        ax.fill_between(
            bin_centers[valid_mask],
            bin_means[valid_mask] - bin_stds[valid_mask],
            bin_means[valid_mask] + bin_stds[valid_mask],
            color="gray", alpha=0.3, zorder=9
        )

    # --- Binned electrode SNR ± std ---
    valid_elec_mask = ~np.isnan(elec_bin_means)
    if np.any(valid_elec_mask):
        ax.plot(elec_bin_centers[valid_elec_mask], elec_bin_means[valid_elec_mask],
                color="black", linestyle="--", linewidth=2.5,
                label="Electrode SNR", zorder=12)
        ax.fill_between(
            elec_bin_centers[valid_elec_mask],
            elec_bin_means[valid_elec_mask] - elec_bin_stds[valid_elec_mask],
            elec_bin_means[valid_elec_mask] + elec_bin_stds[valid_elec_mask],
            color="blue", alpha=0.15, zorder=11
        )

    # --- Plot electrode points ---
    regular_data = {k: v for k, v in electrode_avg.items() if k not in special_names}
    if regular_data:
        reg_dists = [v["dist"] for v in regular_data.values()]
        reg_snrs = [v["snr"] for v in regular_data.values()]
        reg_names = list(regular_data.keys())

        ax.scatter(reg_dists, reg_snrs, s=80, c="gray",
                   edgecolors="gray", linewidths=1, label="Electrodes", zorder=11)

        if show_electrode_labels:
            for name, dist, snr in zip(reg_names, reg_dists, reg_snrs):
                if dist <= x_cap:
                    ax.text(dist, snr + 1, name, ha="center", va="bottom",
                            fontsize=10, alpha=0.7)

    # --- Highlighted electrodes ---
    for name, data in special_data.items():
        if data["dist"] <= x_cap:
            ax.scatter([data["dist"]], [data["snr"]],
                       s=200, marker="D", c="black", edgecolors="black",
                       linewidths=2, zorder=13)
            ax.text(data["dist"], data["snr"] + 1.5, name,
                    ha="center", va="bottom", fontsize=14, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="black", linewidth=1.5))

    # --- Significance markers ---
    y_max = 60
    sig_y_pos = y_max * 0.78
    for i, bc in enumerate(bin_centers):
        if bc <= x_cap:
            pval = pvals_corrected[i] if i < len(pvals_corrected) else np.nan
            if np.isnan(pval):
                ax.text(bc, sig_y_pos, "×", ha="center", va="center",
                        fontsize=26, fontweight="bold", color="gray", alpha=0.6)
            elif significant_bins[i]:
                ax.text(bc, sig_y_pos, "*", ha="center", va="center",
                        fontsize=28, fontweight="bold", color="black")
            else:
                ax.text(bc, sig_y_pos, "n.s.", ha="center", va="center",
                        fontsize=20, color="black")

    # --- Aesthetic & axis formatting ---
    ax.set_xlabel("Distance from S1 hand center (mm)", weight="bold")
    if hemi == 'lh':
        ax.set_ylabel(" SNR (contralateral side)", weight="bold")
    else:
        ax.set_ylabel(" SNR (ipsilateral side)", weight="bold")
        
    ax.set_xlim(0, x_cap)
    ax.set_ylim(0, y_max)

    # Clean spine and tick styling
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.5)

    ax.tick_params(axis="both", length=6, width=1.2, color="black", labelcolor="black")
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    # --- Legend ---
    ax.legend(frameon=False, loc="upper right")

    # --- Save ---
    test_suffix = "one_tailed" if test_type == "one-tailed" else "two_tailed"
    save_path = path.join(
        cwd, f"snr_vs_distance_{hemi}_{rep}_trials_{method}_{test_suffix}.png"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n  Saved: {save_path}")
    print("=" * 60 + "\n")

    return save_path



def compute_electrode_binned_statistics(electrode_avg, bin_size=10.0, max_dist=200.0):
    """
    Compute binned mean and standard deviation for electrode data.
    
    Parameters:
    -----------
    electrode_avg : dict
        Dictionary where keys are electrode names and values are dicts with 'dist' and 'snr'
    bin_size : float
        Width of each bin in mm
    max_dist : float
        Maximum distance to consider
    
    Returns:
    --------
    bin_centers : np.array
        Center positions of bins
    bin_means : np.array
        Mean SNR in each bin
    bin_stds : np.array
        Standard deviation of SNR in each bin
    """
    # Extract distances and SNRs
    dists = np.array([v['dist'] for v in electrode_avg.values()])
    snrs = np.array([v['snr'] for v in electrode_avg.values()])
    
    # Create bins
    bins = np.arange(0, max_dist + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2
    
    # Compute statistics per bin
    bin_means = np.full(len(bin_centers), np.nan)
    bin_stds = np.full(len(bin_centers), np.nan)
    
    for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (dists >= left) & (dists < right)
        bin_data = snrs[mask]
        
        if len(bin_data) > 0:
            bin_means[i] = np.mean(bin_data)
            bin_stds[i] = np.std(bin_data, ddof=1) if len(bin_data) > 1 else 0
    
    return bin_centers, bin_means, bin_stds
#---------------------------------------------------------------------------------------#
def plot_combined_snr_hemispheric(p, method='eLORETA', hand_disk_radius_mm=8.0):
    """
    Main function: analyze and plot SNR vs distance for BOTH hemispheres separately.
    """
    print("\n" + "="*70)
    print(" HEMISPHERIC SNR VS DISTANCE ANALYSIS")
    print("="*70)
    
    subjects = p.subjects
    reps = p.reps
    distance = 5
    cwd = getcwd()
    
    # =================================================================
    # STEP 1: CREATE HAND LABELS FOR BOTH HEMISPHERES
    # =================================================================
    lh_label, lh_center, rh_label, rh_center = make_hand_disk_label_both_hemis(
        subject=p.fs_subject_dir,
        subjects_dir=p.subj_brains_dir,
        parc='HCPMMP1',
        radius_mm=hand_disk_radius_mm
    )
    
    if lh_center is None and rh_center is None:
        print("ERROR: Could not create hand labels for either hemisphere!")
        return
    
    # =================================================================
    # STEP 2: LOAD ALL SUBJECT DATA
    # =================================================================
    _, subject_data = load_subject_data(
        p, subjects, reps, distance, method, cwd
    )
    
    if not subject_data:
        print("ERROR: No subject data loaded!")
        return
    
    n_subjects = len(subject_data)
    print(f"\nProcessing {n_subjects} subjects")

    
    # =================================================================
    # STEP 2.5: CREATE AVERAGE VISUALIZATION PLOTS AND STORE DATA
    # =================================================================
    print(f"\n{'='*70}")
    print(" CREATING AVERAGE SNR VISUALIZATIONS")
    print(f"{'='*70}")
    
    # Dictionary to store averaged data for each repetition
    averaged_data = {}
    
    # Prepare hand disk labels dictionary
    hand_disk_labels = {}
    if lh_label is not None:
        hand_disk_labels['lh'] = lh_label
    if rh_label is not None:
        hand_disk_labels['rh'] = rh_label
    
    # Dictionary to store averaged data for each repetition
    averaged_data = {}
    
    for rep in reps:
        print(f"\n Processing {rep}...")
        
        dist = 15  # or whatever distance you're analyzing
        
        # Create electrode topomap (returns color scale AND averaged electrode data)
        vmin_elec, vmax_elec, averaged_electrode_data = plot_average_electrode_snr_topomap(
            p, subject_data, subjects, rep, cwd, vmin=0, vmax=50
        )
        # DEBUG: Verify what we received
        print(f"\n DEBUG - Received averaged_electrode_data:")
        print(f"   Keys (repetitions): {list(averaged_electrode_data.keys())}")
        if rep in averaged_electrode_data:
            print(f"   Number of channels for {rep}: {len(averaged_electrode_data[rep])}")
            sample_values = list(averaged_electrode_data[rep].values())[:5]
            print(f"   Sample SNR values: {sample_values}")
        
        print(f"\n Electrode color scale: [{vmin_elec:.3f}, {vmax_elec:.3f}]")
        
        # Create source brain plot (returns color scale AND averaged STC)
        # Pass the hand disk labels to be plotted
        vmin_source, vmax_source, averaged_stc = plot_average_source_snr_brain(
            p, subject_data, subjects, rep, method, dist, cwd,
            hand_disk_labels=hand_disk_labels,
            vmin=vmin_elec, vmax=vmax_elec
        )
            
        # Store the averaged data
        averaged_data[rep] = {
            'electrode_data': averaged_electrode_data[rep],
            'source_stc': averaged_stc,
            'vmin': vmin_elec,
            'vmax': vmax_source
        }
        
        if vmin_source is not None:
            print(f" Source brain plot created successfully!")
    
    # =================================================================
    # STEP 3: LOAD SURFACES AND GET ELECTRODE POSITIONS
    # =================================================================
    coords_lh_pial, faces_lh, coords_rh_pial, faces_rh, coords_lh_white = load_surfaces(p)
    
    # Get electrode positions (once, shared)
    template_subj = subjects[0]
    erp_info = subject_data[template_subj]['erp_info']
    electrode_positions, electrode_names, _ = get_electrode_positions(erp_info)
    
    # =================================================================
    # STEP 4: PROCESS EACH HEMISPHERE SEPARATELY USING AVERAGED DATA
    # =================================================================
    for hemi in ['lh', 'rh']:
        print(f"\n{'='*70}")
        print(f" PROCESSING {hemi.upper()} HEMISPHERE")
        print(f"{'='*70}")
    
        center_vert = lh_center if hemi == 'lh' else rh_center
        if center_vert is None:
            print(f" Skipping {hemi.upper()} - no hand center found")
            continue
    
        reference_electrode = 'CP3' if hemi == 'lh' else 'CP4'
        geo_dists_scalp = compute_electrode_distances_from_reference(
            electrode_positions, electrode_names, reference_electrode, hemi
        )
    
        for rep in reps:
            print(f"\n{'-'*70}")
            print(f" {hemi.upper()} - {rep} trials (USING AVERAGED DATA)")
            print(f"{'-'*70}")
    
            averaged_stc = averaged_data[rep]['source_stc']
    
            if hemi == 'lh':
                stc_data = averaged_stc.lh_data
                stc_vertno = averaged_stc.lh_vertno
            else:
                stc_data = averaged_stc.rh_data
                stc_vertno = averaged_stc.rh_vertno
    
            geo_dists, snr_vals = compute_hemisphere_geodesics(
                p, hemi, center_vert, stc_data, stc_vertno, averaged_stc.times
            )
    
            all_subject_hemi_data = [(geo_dists, snr_vals)]
    
            print(f" Using averaged source data: {len(snr_vals)} sources, "
                  f"range [{np.nanmin(snr_vals):.3f}, {np.nanmax(snr_vals):.3f}]")
    
            # Pass the per-repetition electrode averages directly
            elec_dict = averaged_data[rep]['electrode_data']
    
            electrode_avg = collect_hemisphere_electrode_data_from_averaged(
                elec_dict, electrode_names, hemi, geo_dists_scalp, electrode_positions
            )
    
            # Plot normally
            plot_hemisphere_snr_distance(
                hemi, all_subject_hemi_data, electrode_avg,
                rep, method, 1, cwd, bin_size=15.0,
                show_electrode_labels=False,
                label_subset=['CP3', 'CP4', 'CPz'],
                test_type='one-tailed'
            )
 
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE!")
    print("="*70)

#-------------------------------------------------------------------------------------#
def debug_source_data(all_subject_hemi_data, hemi, rep):
    """
    Debug function to investigate source data issues.
    """
    print(f"\n{'='*60}")
    print(f"DEBUG: Investigating {hemi.upper()} source data for {rep}")
    print(f"{'='*60}")
    
    for i, (geo_dists, snr_vals) in enumerate(all_subject_hemi_data):
        print(f"\nSubject {i+1}:")
        print(f"  geo_dists: shape={geo_dists.shape}, dtype={geo_dists.dtype}")
        print(f"    finite: {np.sum(np.isfinite(geo_dists))}/{len(geo_dists)}")
        print(f"    range: [{np.nanmin(geo_dists):.1f}, {np.nanmax(geo_dists):.1f}]")
        
        print(f"  snr_vals: shape={snr_vals.shape}, dtype={snr_vals.dtype}")
        print(f"    finite: {np.sum(np.isfinite(snr_vals))}/{len(snr_vals)}")
        print(f"    range: [{np.nanmin(snr_vals):.6f}, {np.nanmax(snr_vals):.6f}]")
        print(f"    zeros: {np.sum(snr_vals == 0)}")
        print(f"    negative: {np.sum(snr_vals < 0)}")
        
        # Check for matching valid data
        valid_mask = np.isfinite(geo_dists) & np.isfinite(snr_vals)
        print(f"  Both valid: {np.sum(valid_mask)}/{len(valid_mask)}")

#-------------------------------------------------------------------------------------#
def build_cortical_graph(coords, faces):
    """Build adjacency graph for geodesic distance computation on cortical surface."""
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])
    edges = np.vstack([edges, edges[:, ::-1]])  # Symmetrize
    
    edge_lengths = np.linalg.norm(
        coords[edges[:, 0]] - coords[edges[:, 1]], axis=1
    )
    
    graph = coo_matrix(
        (edge_lengths, (edges[:, 0], edges[:, 1])),
        shape=(coords.shape[0], coords.shape[0])
    )
    
    return graph

#-------------------------------------------------------------------------------------#
def compute_surface_normal(coords, faces, vertex_idx):
    """Compute average surface normal at a given vertex."""
    faces_with_vertex = faces[np.any(faces == vertex_idx, axis=1)]
    
    normals = []
    for face in faces_with_vertex:
        v0, v1, v2 = coords[face]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)
        normals.append(normal)
    
    avg_normal = np.mean(normals, axis=0)
    avg_normal = avg_normal / np.linalg.norm(avg_normal)
    
    return avg_normal

#-------------------------------------------------------------------------------------#
def build_scalp_graph(electrode_positions, scalp_reference_point, k=8):
    """Build scalp surface graph using k-nearest neighbors on a spherical surface."""
    all_positions = np.vstack([electrode_positions, scalp_reference_point])
    
    # Fit sphere to electrodes
    center = np.mean(electrode_positions, axis=0)
    radius = np.mean(np.linalg.norm(electrode_positions - center, axis=1))
    print(f"Head sphere: center={center}, radius={radius:.1f}mm")
    
    # Compute pairwise distances
    dist_matrix = distance_matrix(all_positions, all_positions)
    
    # Build k-NN graph
    edges = []
    edge_lengths = []
    
    for i in range(len(all_positions)):
        nearest_indices = np.argpartition(dist_matrix[i], min(k+1, len(all_positions)))[:min(k+1, len(all_positions))]
        nearest_indices = nearest_indices[nearest_indices != i]
        
        for j in nearest_indices:
            if j > i:  # Avoid duplicates
                # Use geodesic distance on sphere (arc length)
                pos_i = all_positions[i] - center
                pos_j = all_positions[j] - center
                
                pos_i_norm = pos_i / np.linalg.norm(pos_i)
                pos_j_norm = pos_j / np.linalg.norm(pos_j)
                
                cos_angle = np.clip(np.dot(pos_i_norm, pos_j_norm), -1, 1)
                angle = np.arccos(cos_angle)
                arc_length = radius * angle
                
                edges.append([i, j])
                edges.append([j, i])  # Symmetric
                edge_lengths.extend([arc_length, arc_length])
    
    edges = np.array(edges)
    edge_lengths = np.array(edge_lengths)
    
    print(f"Created {len(edges)//2} connections using k={k} nearest neighbors")
    
    graph = coo_matrix(
        (edge_lengths, (edges[:, 0], edges[:, 1])),
        shape=(len(all_positions), len(all_positions))
    )
    
    return graph, center, radius

#-------------------------------------------------------------------------------------#
def compute_scalp_geodesic_distances(electrode_positions, scalp_reference_point, k=8):
    """Compute geodesic distances on scalp surface from reference point."""
    print("\n" + "="*60)
    print("Computing scalp geodesic distances")
    print("="*60)
    
    graph, center, radius = build_scalp_graph(electrode_positions, scalp_reference_point, k)
    
    # Reference point is the last element
    ref_idx = len(electrode_positions)
    
    geo_dists_all = dijkstra(csgraph=graph, directed=False, indices=ref_idx)
    geo_dists = geo_dists_all[:-1]  # Exclude reference point itself
    
    n_infinite = np.sum(np.isinf(geo_dists))
    if n_infinite > 0:
        print(f"WARNING: {n_infinite}/{len(geo_dists)} electrodes unreachable!")
        print(f"Retrying with k={k+4}...")
        
        graph, center, radius = build_scalp_graph(electrode_positions, scalp_reference_point, k+4)
        geo_dists_all = dijkstra(csgraph=graph, directed=False, indices=ref_idx)
        geo_dists = geo_dists_all[:-1]
        
        n_infinite = np.sum(np.isinf(geo_dists))
        if n_infinite > 0:
            print(f"Still have {n_infinite} unreachable electrodes")
    
    finite_dists = geo_dists[np.isfinite(geo_dists)]
    if len(finite_dists) > 0:
        print(f"Scalp geodesic distances: [{np.nanmin(finite_dists):.1f}, "
              f"{np.nanmax(finite_dists):.1f}] mm")
        print(f"Valid distances: {len(finite_dists)}/{len(geo_dists)}")
    
    return geo_dists

#-------------------------------------------------------------------------------------#
def get_electrode_positions(erp_info):
    """Extract electrode positions from ERP info."""
    montage = mne.channels.make_standard_montage("standard_1005")
    erp_info_copy = erp_info.copy()
    erp_info_copy.set_montage(montage, match_case=False, on_missing='ignore')
    
    electrode_positions = []
    electrode_names = []
    
    for ch in erp_info_copy['chs']:
        pos = ch['loc'][:3] * 1000  # Convert to mm
        if not np.allclose(pos, 0):
            electrode_positions.append(pos)
            electrode_names.append(ch['ch_name'])
    
    electrode_positions = np.array(electrode_positions)
    print(f"Found {len(electrode_positions)} electrode positions")
    
    return electrode_positions, electrode_names, erp_info_copy

#-------------------------------------------------------------------------------------#
def load_subject_data(p, subjects, reps, distance, method, cwd):
    """Load electrode and source SNR data for all subjects."""
    print("\n" + "="*50)
    print("Loading data for all subjects...")
    print("="*50)
    
    subject_data = {}
    
    for subj in subjects:
        print(f"\nLoading data for subject: {subj}")
        subj_number = ''.join(filter(str.isdigit, subj))
        
        # File paths
        snr_file = f"snr_{subj_number}_el.csv"
        snr_path = path.join(cwd, p.data, p.processed_data, p.protocol, subj, 
                             p.session, p.current_dir, snr_file)
        erp_path = path.join(cwd, p.data, p.processed_data, p.protocol, subj, 
                             p.session, p.eeg_dir, p.epochsfile)
        stc_dir = path.join(cwd, p.data, p.processed_data, p.protocol, subj, 
                            p.session, p.current_dir)
        
        if not path.exists(erp_path) or not path.exists(snr_path):
            print(f"Missing files for {subj}. Skipping.")
            continue
        
        # Load ERP info
        try:
            epochs = mne.read_epochs(erp_path, preload=False)
            erp_info = epochs.info.copy()
            del epochs
        except Exception as e:
            print(f"Could not load ERP for {subj}: {e}")
            continue
        
        # Load electrode SNR
        df_el = pd.read_csv(snr_path)
        
        # Load source SNR for all reps
        source_snr = {}
        for rep in reps:
            stc_file = f"current{rep}reps_{distance}mm_{method}-lh.stc"
            stc_path = path.join(stc_dir, stc_file)
            
            if path.exists(stc_path):
                try:
                    stc = mne.read_source_estimate(stc_path)
                    
                    print(f"  Loaded {rep} STC: data shape = {stc.data.shape}")
                    print(f"    LH vertices: {len(stc.vertices[0])}, RH vertices: {len(stc.vertices[1])}")
                    print(f"    Data range: [{np.nanmin(stc.data):.6f}, {np.nanmax(stc.data):.6f}]")
                    
                    # Store the raw STC
                    source_snr[rep] = {'stc': stc}
                    
                except Exception as e:
                    print(f" ERROR loading STC: {e}")
                    source_snr[rep] = None
            else:
                print(f"  STC file not found: {stc_file}")
                source_snr[rep] = None
        
        subject_data[subj] = {
            'electrode_snr': df_el,
            'source_snr': source_snr,
            'erp_info': erp_info
        }
    
    print(f"\nSuccessfully loaded data for {len(subject_data)} subjects")
    return None, subject_data  # Return None for first value since we don't use it
#-------------------------------------------------------------------------------------#
def load_surfaces(p):
    """Load cortical surfaces (pial and white)."""
    coords_lh_pial, faces_lh = mne.read_surface(
        path.join(p.subj_brains_dir, p.fs_subject_dir, "surf", "lh.pial"))
    coords_rh_pial, faces_rh = mne.read_surface(
        path.join(p.subj_brains_dir, p.fs_subject_dir, "surf", "rh.pial"))
    coords_lh_white, _ = mne.read_surface(
        path.join(p.subj_brains_dir, p.fs_subject_dir, "surf", "lh.white"))
    
    print(f"Surfaces loaded: LH pial={len(coords_lh_pial)}, "
          f"RH pial={len(coords_rh_pial)}, LH white={len(coords_lh_white)} vertices")
    
    if len(coords_lh_white) != len(coords_lh_pial):
        print("WARNING: White and pial surfaces have different vertex counts!")
    
    return coords_lh_pial, faces_lh, coords_rh_pial, faces_rh, coords_lh_white

#-------------------------------------------------------------------------------------#
def make_hand_disk_label(subject, subjects_dir, parc='HCPMMP1', radius_mm=8.0):
        """
        Build a small disk-like label (vertices within radius_mm of the centroid)
        using HCPMMP1 left-hemisphere S1-related labels (3b,1,2).
        Returns (mne.Label, center_vertex_index) or (None, None) if not found.
        """
        print(f"\n[make_hand_disk_label] Building hand disk for subject '{subject}' | radius={radius_mm} mm")

        # Load annotation
        try:
            labels = mne.read_labels_from_annot(subject=subject,
                                                parc=parc,
                                                hemi='both',
                                                subjects_dir=subjects_dir)
        except Exception as e:
            print(f" Could not read annotation '{parc}' for {subject} in {subjects_dir}: {e}")
            return None, None

        # Extract S1-related labels
        s1_patterns = ['L_3b_ROI-lh', 'L_1_ROI-lh', 'L_2_ROI-lh']
        left_labels = [lbl for lbl in labels if lbl.name in s1_patterns]

        if not left_labels:
            left_labels = [lbl for lbl in labels if lbl.name.startswith('L_') and
                           any(x in lbl.name for x in ['3b', '_1_', '_2_'])]

        if not left_labels:
            print(f"  No left-hemisphere S1 labels found in {parc} for {subject}.")
            return None, None

        # Combine vertices from those labels
        left_vertices = np.unique(np.hstack([lbl.vertices for lbl in left_labels]))

        # Load surface and find centroid
        surf_fname = path.join(subjects_dir, subject, 'surf', 'lh.white')
        try:
            coords, faces = mne.read_surface(surf_fname)
        except Exception as e:
            print(f" Could not read surface '{surf_fname}': {e}")
            return None, None

        # Compute centroid and radius-based disk
        centroid = coords[left_vertices].mean(axis=0)
        dists = np.linalg.norm(coords - centroid, axis=1)
        selected_vertices = np.where(dists <= radius_mm)[0]

        if selected_vertices.size == 0:
            print(f" No vertices found within {radius_mm} mm of centroid — try increasing the radius.")
            return None, None

        # Build label
        try:
            disk_label = Label(selected_vertices, pos=coords[selected_vertices],
                               hemi='lh', name='S1_hand_disk', subject=subject)
        except Exception:
            disk_label = Label(selected_vertices, pos=coords[selected_vertices])
            disk_label.hemi = 'lh'
            disk_label.name = 'S1_hand_disk'
            disk_label.subject = subject

        center_vertex = int(np.argmin(dists))
        print(f" Hand disk built: {len(selected_vertices)} vertices | center vertex {center_vertex}")
        return disk_label, center_vertex

#-----------------------------------------------------------------------------------------#
def compute_electrode_distances_from_reference(electrode_positions, electrode_names, 
                                               reference_electrode, hemi, offset_mm=20.0):
    """
    Compute geodesic distances on scalp from a reference electrode (CP3/CP4).
    Add a fixed offset so distances don't start at zero.
    
    Parameters
    ----------
    electrode_positions : array
        N x 3 array of electrode positions
    electrode_names : list
        List of electrode names
    reference_electrode : str
        Reference electrode name ('CP3' for LH, 'CP4' for RH)
    hemi : str
        'lh' or 'rh'
    offset_mm : float
        Fixed offset to add to all distances (default: 60mm)
        This represents the approximate distance from cortical S1 to scalp
        
    Returns
    -------
    geo_dists : array
        Geodesic distances from reference electrode + offset
    """
    print(f"\n[compute_electrode_distances_from_reference] Using {reference_electrode} as reference for {hemi.upper()}")
    
    # Find reference electrode
    try:
        ref_idx = electrode_names.index(reference_electrode)
        ref_pos = electrode_positions[ref_idx]
        print(f" Reference electrode position: {ref_pos}")
    except ValueError:
        print(f" WARNING: {reference_electrode} not found! Using centroid instead.")
        ref_pos = np.mean(electrode_positions, axis=0)
        ref_idx = None
    
    # Build scalp graph
    graph, center, radius = build_scalp_graph_electrodes_only(
        electrode_positions, ref_pos, k=8
    )
    
    # Compute geodesic distances
    if ref_idx is not None:
        geo_dists_all = dijkstra(csgraph=graph, directed=False, indices=ref_idx)
    else:
        # If reference not found, compute from added reference point (last index)
        graph, center, radius = build_scalp_graph_with_reference(
            electrode_positions, ref_pos, k=8
        )
        geo_dists_all = dijkstra(csgraph=graph, directed=False, indices=len(electrode_positions))
        geo_dists_all = geo_dists_all[:-1]  # Remove reference point itself
    
    # Check for unreachable electrodes
    n_infinite = np.sum(np.isinf(geo_dists_all))
    if n_infinite > 0:
        print(f" WARNING: {n_infinite} unreachable electrodes, retrying with k=12")
        graph, center, radius = build_scalp_graph_electrodes_only(
            electrode_positions, ref_pos, k=12
        )
        if ref_idx is not None:
            geo_dists_all = dijkstra(csgraph=graph, directed=False, indices=ref_idx)
        else:
            graph, center, radius = build_scalp_graph_with_reference(
                electrode_positions, ref_pos, k=12
            )
            geo_dists_all = dijkstra(csgraph=graph, directed=False, indices=len(electrode_positions))
            geo_dists_all = geo_dists_all[:-1]
    
    # ADD THE OFFSET TO ALL DISTANCES
    geo_dists_with_offset = geo_dists_all + offset_mm
    
    finite_dists = geo_dists_with_offset[np.isfinite(geo_dists_with_offset)]
    if len(finite_dists) > 0:
        print(f" Electrode distances (before offset): [{np.nanmin(geo_dists_all[np.isfinite(geo_dists_all)]):.1f}, "
              f"{np.nanmax(geo_dists_all[np.isfinite(geo_dists_all)]):.1f}] mm")
        print(f" Electrode distances (after +{offset_mm}mm offset): [{np.nanmin(finite_dists):.1f}, {np.nanmax(finite_dists):.1f}] mm")
        print(f" Valid: {len(finite_dists)}/{len(geo_dists_with_offset)}")
        
        # Verify reference electrode is at the offset
        if ref_idx is not None:
            print(f" {reference_electrode} is now at: {geo_dists_with_offset[ref_idx]:.1f}mm")
    
    return geo_dists_with_offset


def build_scalp_graph_electrodes_only(electrode_positions, ref_pos, k=8):
    """
    Build scalp surface graph using only electrode positions.
    Simpler version that doesn't add extra reference point if not needed.
    """
    # Fit sphere to electrodes
    center = np.mean(electrode_positions, axis=0)
    radius = np.mean(np.linalg.norm(electrode_positions - center, axis=1))
    print(f" Head sphere: center={center}, radius={radius:.1f}mm")
    
    # Use all positions
    all_positions = electrode_positions
    
    # Compute pairwise distances
    dist_matrix = distance_matrix(all_positions, all_positions)
    
    # Build k-NN graph
    edges = []
    edge_lengths = []
    
    for i in range(len(all_positions)):
        # Get k+1 nearest neighbors (includes self)
        nearest_indices = np.argpartition(dist_matrix[i], min(k+1, len(all_positions)))[:min(k+1, len(all_positions))]
        nearest_indices = nearest_indices[nearest_indices != i]
        
        for j in nearest_indices:
            if j > i:  # Avoid duplicates
                # Use geodesic distance on sphere (arc length)
                pos_i = all_positions[i] - center
                pos_j = all_positions[j] - center
                
                pos_i_norm = pos_i / np.linalg.norm(pos_i)
                pos_j_norm = pos_j / np.linalg.norm(pos_j)
                
                cos_angle = np.clip(np.dot(pos_i_norm, pos_j_norm), -1, 1)
                angle = np.arccos(cos_angle)
                arc_length = radius * angle
                
                edges.append([i, j])
                edges.append([j, i])  # Symmetric
                edge_lengths.extend([arc_length, arc_length])
    
    edges = np.array(edges)
    edge_lengths = np.array(edge_lengths)
    
    print(f" Created {len(edges)//2} connections using k={k}")
    
    graph = coo_matrix(
        (edge_lengths, (edges[:, 0], edges[:, 1])),
        shape=(len(all_positions), len(all_positions))
    )
    
    return graph, center, radius
#--------------------------------------------------------------------------------#

def compute_electrode_distances_with_offset(electrode_positions, electrode_names, 
                                           cortical_hand_center, cortical_normal,
                                           reference_electrode, hemi):
    """
    Compute electrode distances with anatomical offset correction.
    
    The offset is the distance from the cortical hand center (projected to scalp)
    to the reference electrode (CP3/CP4).
    
    Parameters
    ----------
    cortical_hand_center : array
        3D position of hand center on cortex
    cortical_normal : array
        Normal vector at hand center
    reference_electrode : str
        'CP3' for LH, 'CP4' for RH
        
    Returns
    -------
    geo_dists_corrected : array
        Geodesic distances corrected for anatomical offset
    scalp_projection_point : array
        Where the normal intersects the scalp
    offset_distance : float
        Distance from projection to reference electrode
    """
    print(f"\n[compute_electrode_distances_with_offset] {hemi.upper()} hemisphere")
    
    # Project cortical center to scalp surface
    # Find intersection with electrode "sphere"
    electrode_center = np.mean(electrode_positions, axis=0)
    electrode_radius = np.mean(np.linalg.norm(electrode_positions - electrode_center, axis=1))
    
    print(f" Electrode sphere: center={electrode_center}, radius={electrode_radius:.1f}mm")
    
    # Project along normal until we hit the electrode radius
    # Start from cortical center and move along normal
    projection_distance = 50.0  # mm, initial guess
    scalp_projection_point = cortical_hand_center + cortical_normal * projection_distance
    
    # Refine to actually hit the sphere
    for _ in range(10):  # Iterate to find intersection
        dist_from_center = np.linalg.norm(scalp_projection_point - electrode_center)
        if np.abs(dist_from_center - electrode_radius) < 0.1:  # Within 0.1mm
            break
        # Adjust projection distance
        correction = electrode_radius - dist_from_center
        projection_distance += correction
        scalp_projection_point = cortical_hand_center + cortical_normal * projection_distance
    
    print(f" Scalp projection point: {scalp_projection_point}")
    print(f" Projection distance from cortex: {projection_distance:.1f}mm")
    
    # Find reference electrode position
    try:
        ref_idx = electrode_names.index(reference_electrode)
        ref_pos = electrode_positions[ref_idx]
        
        # Calculate offset (Euclidean distance on scalp)
        offset_distance = np.linalg.norm(scalp_projection_point - ref_pos)
        print(f" Offset from projection to {reference_electrode}: {offset_distance:.1f}mm")
    except ValueError:
        print(f" WARNING: {reference_electrode} not found!")
        ref_idx = None
        ref_pos = scalp_projection_point
        offset_distance = 0.0
    
    # Compute geodesic distances from reference electrode
    if ref_idx is not None:
        graph, _, _ = build_scalp_graph_electrodes_only(electrode_positions, ref_pos, k=8)
        geo_dists_raw = dijkstra(csgraph=graph, directed=False, indices=ref_idx)
        
        # Check for unreachable
        n_infinite = np.sum(np.isinf(geo_dists_raw))
        if n_infinite > 0:
            print(f" WARNING: {n_infinite} unreachable, retrying with k=12")
            graph, _, _ = build_scalp_graph_electrodes_only(electrode_positions, ref_pos, k=12)
            geo_dists_raw = dijkstra(csgraph=graph, directed=False, indices=ref_idx)
    else:
        geo_dists_raw = np.full(len(electrode_positions), np.nan)
    
    # Apply offset correction: distances are relative to the projected hand center
    geo_dists_corrected = geo_dists_raw + offset_distance
    
    finite_dists = geo_dists_corrected[np.isfinite(geo_dists_corrected)]
    if len(finite_dists) > 0:
        print(f" Corrected distances: [{np.nanmin(finite_dists):.1f}, {np.nanmax(finite_dists):.1f}] mm")
        print(f" Valid: {len(finite_dists)}/{len(geo_dists_corrected)}")
    
    return geo_dists_corrected, scalp_projection_point, offset_distance

#---------------------------------------------------------------------------------------#
'''
def compute_significance_source_vs_electrode(bin_centers, all_subject_source_data, 
                                            electrode_avg, alpha=0.05):
    """
    Compare ALL source points vs ALL electrode measurements within each distance bin.
    
    For each bin:
    - Collects all individual source points (from all subjects) in that distance range
    - Collects all electrode measurements in that distance range
    - Tests if these two distributions differ significantly
    
    Parameters
    ----------
    bin_centers : array
        Distance bin centers
    all_subject_source_data : list of tuples
        [(geo_dists_subj1, snr_values_subj1), (geo_dists_subj2, snr_values_subj2), ...]
    electrode_avg : dict
        {electrode_name: {'snr': value, 'dist': distance, 'n': count}}
    alpha : float
        FDR significance level
        
    Returns
    -------
    significant_bins : array of bool
        True where source SNR significantly differs from electrode SNR
    """
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    
    print(f"\n[compute_significance_source_vs_electrode] Comparing source vs electrode at each distance")
    
    # Determine bin size
    bin_size = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 20.0
    
    p_values = []
    test_results = []
    
    for i, bc in enumerate(bin_centers):
        bin_start = bc - bin_size / 2
        bin_end = bc + bin_size / 2
        
        # ========================================
        # 1. Collect ALL individual source points in this bin
        # ========================================
        source_points_in_bin = []
        
        for subj_idx, (geo_dists, snr_values) in enumerate(all_subject_source_data):
            # Ensure arrays
            geo_dists = np.asarray(geo_dists).flatten()
            snr_values = np.asarray(snr_values).flatten()
            
            # Filter valid data
            valid = np.isfinite(geo_dists) & np.isfinite(snr_values)
            geo_dists_valid = geo_dists[valid]
            snr_valid = snr_values[valid]
            
            # Find points in this bin
            in_bin = (geo_dists_valid >= bin_start) & (geo_dists_valid < bin_end)
            
            # Add all individual points
            source_points_in_bin.extend(snr_valid[in_bin])
        
        source_data = np.array(source_points_in_bin)
        
        # ========================================
        # 2. Collect ALL electrode measurements in this bin
        # ========================================
        electrode_points_in_bin = []
        electrode_names_in_bin = []
        
        for elec_name, elec_info in electrode_avg.items():
            elec_dist = elec_info['dist']
            
            if bin_start <= elec_dist < bin_end:
                # Get electrode SNR
                elec_snr = elec_info['snr']
                
                # If 'n' represents number of measurements, include that many copies
                # Otherwise, just include once
                n_measurements = elec_info.get('n', 1)
                
                electrode_points_in_bin.extend([elec_snr] * n_measurements)
                electrode_names_in_bin.append(elec_name)
        
        electrode_data = np.array(electrode_points_in_bin)
        
        # ========================================
        # 3. Statistical test
        # ========================================
        
        # Need sufficient data for both groups
        min_source = 10  # At least 10 source points
        min_electrode = 2  # At least 5 electrode measurements
        
        if len(source_data) < min_source or len(electrode_data) < min_electrode:
            p_values.append(1.0)
            test_results.append({
                'bin': bc,
                'bin_range': (bin_start, bin_end),
                'n_source': len(source_data),
                'n_electrode': len(electrode_data),
                'n_electrodes': len(electrode_names_in_bin),
                'source_mean': np.nan if len(source_data) == 0 else np.mean(source_data),
                'electrode_mean': np.nan if len(electrode_data) == 0 else np.mean(electrode_data),
                'p_value': 1.0,
                'test': 'insufficient_data'
            })
            continue
        
        # Compute statistics
        source_mean = np.mean(source_data)
        source_std = np.std(source_data)
        electrode_mean = np.mean(electrode_data)
        electrode_std = np.std(electrode_data)
        
        # Mann-Whitney U test (non-parametric, independent samples)
        try:
            statistic, p_val = mannwhitneyu(source_data, electrode_data, 
                                           alternative='two-sided')
            
            p_values.append(p_val)
            
            test_results.append({
                'bin': bc,
                'bin_range': (bin_start, bin_end),
                'n_source': len(source_data),
                'n_electrode': len(electrode_data),
                'n_electrodes': len(electrode_names_in_bin),
                'electrodes': electrode_names_in_bin,
                'source_mean': source_mean,
                'source_std': source_std,
                'electrode_mean': electrode_mean,
                'electrode_std': electrode_std,
                'difference': source_mean - electrode_mean,
                'effect_size': (source_mean - electrode_mean) / np.sqrt((source_std**2 + electrode_std**2) / 2),
                'p_value': p_val,
                'U_statistic': statistic,
                'test': 'mann_whitney'
            })
            
        except Exception as e:
            print(f"  Bin {i} ({bc:.0f}mm): Test failed: {e}")
            p_values.append(1.0)
            test_results.append({
                'bin': bc,
                'bin_range': (bin_start, bin_end),
                'n_source': len(source_data),
                'n_electrode': len(electrode_data),
                'source_mean': source_mean,
                'electrode_mean': electrode_mean,
                'p_value': 1.0,
                'test': 'failed',
                'error': str(e)
            })
    
    # ========================================
    # 4. FDR Correction
    # ========================================
    p_values = np.array(p_values)
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    # ========================================
    # 5. Report Results
    # ========================================
    n_tested = sum(1 for r in test_results if r['test'] == 'mann_whitney')
    n_sig = np.sum(reject)
    
    print(f"\n Statistical Results:")
    print(f"  Bins tested: {n_tested}/{len(bin_centers)}")
    print(f"  Significant differences (FDR q<{alpha}): {n_sig}")
    
    # Show sample sizes
    tested_results = [r for r in test_results if r['test'] == 'mann_whitney']
    if tested_results:
        source_ns = [r['n_source'] for r in tested_results]
        electrode_ns = [r['n_electrode'] for r in tested_results]
        print(f"\n  Sample sizes per bin:")
        print(f"    Source points: min={min(source_ns)}, max={max(source_ns)}, mean={np.mean(source_ns):.1f}")
        print(f"    Electrode points: min={min(electrode_ns)}, max={max(electrode_ns)}, mean={np.mean(electrode_ns):.1f}")
    
    if n_sig > 0:
        print(f"\n  Significant bins:")
        for i, (is_sig, result) in enumerate(zip(reject, test_results)):
            if is_sig and result['test'] == 'mann_whitney':
                diff = result['difference']
                direction = "SOURCE > ELECTRODE" if diff > 0 else "ELECTRODE > SOURCE"
                print(f"    {result['bin']:.0f}mm ({result['bin_range'][0]:.0f}-{result['bin_range'][1]:.0f}mm): {direction}")
                print(f"      Difference: Δ={diff:.2f}, Effect size={result['effect_size']:.2f}")
                print(f"      p-value: {result['p_value']:.4f} → p_corrected: {pvals_corrected[i]:.4f}")
                print(f"      Source: {result['source_mean']:.2f}±{result['source_std']:.2f} (N={result['n_source']} points)")
                print(f"      Electrode: {result['electrode_mean']:.2f}±{result['electrode_std']:.2f} (N={result['n_electrode']} points from {result['n_electrodes']} electrodes)")
                if result['electrodes']:
                    elec_list = ', '.join(result['electrodes'][:8])
                    if len(result['electrodes']) > 8:
                        elec_list += f" ... (+{len(result['electrodes'])-8} more)"
                    print(f"      Electrodes: {elec_list}")
    else:
        print(f"\n  No significant differences found.")
        print(f"  Showing p-values for first 5 tested bins:")
        for i, result in enumerate(test_results[:5]):
            if result['test'] == 'mann_whitney':
                print(f"    {result['bin']:.0f}mm: p={result['p_value']:.4f}, "
                      f"diff={result['difference']:.2f} "
                      f"(N_source={result['n_source']}, N_elec={result['n_electrode']})")
    
    # Summary by distance range
    print(f"\n  Distance ranges:")
    for dist_range, label in [((0, 100), "0-100mm"), 
                               ((100, 150), "100-150mm"), 
                               ((150, 250), "150-250mm")]:
        bins_in_range = [(i, r) for i, r in enumerate(test_results) 
                        if dist_range[0] <= r['bin'] < dist_range[1] 
                        and r['test'] == 'mann_whitney']
        if bins_in_range:
            n_sig_range = sum(1 for i, r in bins_in_range if reject[i])
            avg_diff = np.mean([r['difference'] for i, r in bins_in_range])
            avg_source = np.mean([r['source_mean'] for i, r in bins_in_range])
            avg_electrode = np.mean([r['electrode_mean'] for i, r in bins_in_range])
            print(f"    {label}: {n_sig_range}/{len(bins_in_range)} significant, "
                  f"avg diff={avg_diff:.2f} (source={avg_source:.2f}, electrode={avg_electrode:.2f})")
    
    return reject
'''
#-----------------------------------------------------------------------------------------------#
def plot_average_electrode_snr_topomap(p, subject_data, subjects, rep, cwd, vmin=None, vmax=None):
    """
    Plot average electrode SNR topomaps across subjects and repetitions.
    Returns the averaged electrode SNR data for use in distance plots.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    print("\n[plot_average_electrode_snr_topomap] Creating electrode SNR topomaps")

    all_data_full = []
    channels_of_interest = ["CP3", "CPz", "CP4"]

    # Load per-subject electrode SNR CSVs
    for subj in subjects:
        subj_number = ''.join(filter(str.isdigit, subj))
        filename = f"snr_{subj_number}_el.csv"
        filepath = os.path.join(
            cwd, p.data, p.processed_data, p.protocol,
            subj, p.session, p.current_dir, filename
        )

        if not os.path.exists(filepath):
            print(f" Missing SNR file for {subj}. Skipping.")
            continue

        df = pd.read_csv(filepath)
        df["Subject"] = subj
        all_data_full.append(df)

    if not all_data_full:
        print(" No electrode SNR data found. Exiting.")
        return None, None, None

    df_all = pd.concat(all_data_full, ignore_index=True)
    repetition_numbers = sorted(df_all["n_select"].unique())
    all_channels = sorted(df_all["channel"].unique())

    # Compute mean SNR per channel across subjects for each repetition
    snr_by_trial = (
        df_all.groupby(["n_select", "channel"])["snr_mean"]
        .mean()
        .unstack(fill_value=np.nan)
    )
    
    # DEBUG: Check the computed SNR values
    print(f"\n DEBUG - Averaged electrode SNR values:")
    print(f"   Shape: {snr_by_trial.shape}")
    print(f"   Repetitions: {snr_by_trial.index.tolist()}")
    print(f"   Channels (first 10): {snr_by_trial.columns.tolist()[:10]}")
    print(f"   Value range: [{snr_by_trial.min().min():.3f}, {snr_by_trial.max().max():.3f}]")

    # Create MNE info structure for topomap
    montage = mne.channels.make_standard_montage("standard_1005")
    info = mne.create_info(ch_names=all_channels, sfreq=1000, ch_types="eeg")
    info.set_montage(montage)

    # Determine color scale
    if vmin is None or vmax is None:
        all_snr_values = snr_by_trial.values.flatten()
        all_snr_values = all_snr_values[~np.isnan(all_snr_values)]
        vmin = np.percentile(all_snr_values, 1)
        vmax = np.percentile(all_snr_values, 99)
        print(f" Color scale (shared): [{vmin:.2f}, {vmax:.2f}]")

    # Plot each repetition's topomap
    for n_trials in repetition_numbers:
        snr_for_trials = snr_by_trial.loc[n_trials].reindex(all_channels)
        snr_values = snr_for_trials.values

        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot topomap
        plot_result = mne.viz.plot_topomap(
            snr_values,
            info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            contours=6,
            vlim=(vmin, vmax)
        )
        
        # Handle contour thickness robustly
        if isinstance(plot_result, tuple) and len(plot_result) > 1:
            im, contour_obj = plot_result
            try:
                # Works in most MNE versions
                for coll in getattr(contour_obj, 'collections', []):
                    coll.set_linewidth(1.5)
            except Exception as e:
                print(f"Warning: Could not set contour line width ({type(contour_obj)}): {e}")
        else:
            im = plot_result  # if only one object returned
        
        # Title
        ax.set_title(f"Average SNR – {n_trials} Trials", fontsize=14, fontweight="bold")
        
        # Colorbar with larger fonts and thicker lines
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("SNR", rotation=270, labelpad=25, fontsize=20)
        cbar.ax.tick_params(labelsize=18, width=1.8, length=6)
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(1.5)
        
        plt.tight_layout()

        save_path = os.path.join(cwd, f"topoplot_snr_{n_trials}_trials.png")
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close()
        print(f" Saved topomap for {n_trials} trials → {save_path}")

    # Return the averaged data as a dictionary: {repetition: {channel: snr_value}}
    averaged_electrode_data = {}
    for n_trials in repetition_numbers:
        snr_for_trials = snr_by_trial.loc[n_trials].reindex(all_channels)
        averaged_electrode_data[n_trials] = snr_for_trials.to_dict()
        
        # DEBUG: Show what we're returning
        sample_channels = ['CP3', 'CPz', 'CP4']
        print(f"\n DEBUG - Returning data for {n_trials} trials:")
        for ch in sample_channels:
            if ch in averaged_electrode_data[n_trials]:
                print(f"   {ch}: {averaged_electrode_data[n_trials][ch]:.3f}")

    print("\n All electrode SNR topomaps generated successfully.")
    return vmin, vmax, averaged_electrode_data


def plot_average_source_snr_brain(p, subject_data, subjects, rep, method, dist, cwd, 
                                   hand_disk_labels=None, vmin=None, vmax=None):
    """
    Plot average source SNR across all subjects on brain surface with sensory cortex labels.
    Uses the actual computed SNR values from CSV files, NOT raw amplitudes.
    
    Parameters
    ----------
    p : params object
        Analysis parameters
    subject_data : dict
        Dictionary containing all subject data
    subjects : list
        List of subject IDs
    rep : str or int
        Trial repetition (e.g., 125, 250, 500, etc.)
    method : str
        Source localization method (e.g., 'dSPM', 'eLORETA')
    dist : str or int
        Distance in mm (e.g., 20)
    cwd : str
        Current working directory
    hand_disk_labels : dict, optional
        Dictionary with keys 'lh' and 'rh' containing hand disk Label objects
    vmin, vmax : float
        Color scale limits (shared with electrode plot)
    """
    import pandas as pd
    import numpy as np
    import mne
    from os import path
    
    print(f"\n[plot_average_source_snr_brain] Creating source SNR brain plot for {rep}")
    
    # Convert rep to string if needed for consistency
    rep_str = str(rep)
    
    # Load a template STC file to get vertex structure
    # Use the first subject's STC file
    template_subj = subjects[0]
    subj_number = ''.join(filter(str.isdigit, template_subj))
    
    # Construct path to template STC file
    stc_filename = f"current{rep_str}reps_{dist}mm_{method}-lh.stc"
    stc_path = path.join(
        cwd, p.data, p.processed_data, p.protocol,
        template_subj, p.session, p.current_dir, stc_filename
    )
    
    if not path.exists(stc_path):
        print(f" Template STC file not found: {stc_path}")
        return None, None
    
    template_stc = mne.read_source_estimate(stc_path)
    
    print(f" Template STC structure (from {template_subj}):")
    print(f"   LH vertices: {len(template_stc.lh_vertno)}")
    print(f"   RH vertices: {len(template_stc.rh_vertno)}")
    print(f"   Total vertices: {template_stc.data.shape[0]}")
    
    # Load and average SNR values from CSV files (the actual computed SNR!)
    all_snr_arrays = []
    
    for subj in subjects:
        subj_number = ''.join(filter(str.isdigit, subj))
        
        # Construct path to SNR CSV file
        # Format: snr_dSPM_20mm_source.csv
        csv_filename = f"snr_{method}_{dist}mm_source.csv"
        csv_path = path.join(
            cwd, p.data, p.processed_data, p.protocol,
            subj, p.session, p.current_dir, csv_filename
        )
        
        if not path.exists(csv_path):
            print(f" Warning: SNR CSV not found for {subj}: {csv_path}")
            continue
        
        # Load SNR data
        df = pd.read_csv(csv_path)
        
        # Filter for this specific repetition
        df = df[df['rep'] == int(rep_str)]
        
        if len(df) == 0:
            print(f" Warning: No data for rep {rep_str} in {subj}")
            continue
        
        # Sort by source index to ensure correct vertex order
        df = df.sort_values('source')
        snr_values = df['snr'].values
        
        # Verify we have the right number of sources
        if len(snr_values) != template_stc.data.shape[0]:
            print(f" Warning: Source count mismatch for {subj}: "
                  f"expected {template_stc.data.shape[0]}, got {len(snr_values)}")
            continue
        
        all_snr_arrays.append(snr_values)
        print(f" Loaded SNR data for {subj}: {len(snr_values)} sources, "
              f"range [{np.min(snr_values):.3f}, {np.max(snr_values):.3f}]")
    
    if not all_snr_arrays:
        print(f" ERROR: No SNR data found for any subject")
        return None, None
    
    print(f" Collected SNR data from {len(all_snr_arrays)} subjects")
    
    # Average SNR across subjects
    avg_snr = np.mean(np.stack(all_snr_arrays, axis=0), axis=0)
    
    print(f" Averaged SNR statistics:")
    print(f"   Shape: {avg_snr.shape}")
    print(f"   Range: [{np.min(avg_snr):.3f}, {np.max(avg_snr):.3f}]")
    print(f"   Mean: {np.mean(avg_snr):.3f}")
    print(f"   Median: {np.median(avg_snr):.3f}")
    print(f"   Non-zero values: {np.sum(avg_snr > 0)}/{len(avg_snr)}")
    
    # Split into LH and RH
    n_lh = len(template_stc.lh_vertno)
    avg_lh_snr = avg_snr[:n_lh]
    avg_rh_snr = avg_snr[n_lh:]
    
    print(f" LH SNR range: [{np.min(avg_lh_snr):.3f}, {np.max(avg_lh_snr):.3f}]")
    print(f" RH SNR range: [{np.min(avg_rh_snr):.3f}, {np.max(avg_rh_snr):.3f}]")
    
    # Determine color scale
    # Use the actual data range so only the best areas show as red
    source_vmax = np.max(avg_snr)
    source_vmin = np.min(avg_snr)
    
    if vmin is None or vmax is None:
        # Use actual min and max to show full dynamic range
        vmin = source_vmin
        vmax = source_vmax
        print(f" Using full source data range: [{vmin:.3f}, {vmax:.3f}]")
    else:
        # Override with actual source max so only peak areas are red
        print(f" Electrode scale provided: [{vmin:.3f}, {vmax:.3f}]")
        print(f" Overriding to use actual source max: {source_vmax:.3f}")
        vmax = source_vmax
        # Keep electrode vmin or use source vmin, whichever is more appropriate
        vmin = min(vmin, source_vmin)
        print(f" Final color scale: [{vmin:.3f}, {vmax:.3f}]")
    
    print(f" Source SNR range: [{source_vmin:.3f}, {source_vmax:.3f}]")
    
    # Create STC with SNR values
    # Need to expand to time dimension for MNE plotting
    n_timepoints = len(template_stc.times)
    snr_data_2d = np.repeat(avg_snr[:, np.newaxis], n_timepoints, axis=1)
    
    # Create averaged STC
    avg_stc = mne.SourceEstimate(
        data=snr_data_2d,
        vertices=[template_stc.lh_vertno, template_stc.rh_vertno],
        tmin=template_stc.tmin,
        tstep=template_stc.tstep,
        subject=p.fs_subject_dir
    )
    
    # Load sensory cortex labels
    try:
        labels = mne.read_labels_from_annot(
            subject=p.fs_subject_dir,
            parc='HCPMMP1',
            hemi='both',
            subjects_dir=p.subj_brains_dir
        )
        
        # Get S1 labels (areas 3b, 1, 2 for both hemispheres)
        s1_patterns = ['L_3b_ROI-lh', 'L_1_ROI-lh', 'L_2_ROI-lh',
                       'R_3b_ROI-rh', 'R_1_ROI-rh', 'R_2_ROI-rh']
        s1_labels = [lbl for lbl in labels if lbl.name in s1_patterns]
        
        print(f" Found {len(s1_labels)} sensory cortex labels")
    except Exception as e:
        print(f" Could not load labels: {e}")
        s1_labels = None
    
    # Plot brain WITHOUT built-in colorbar (we'll add our own)
    brain = avg_stc.plot(
        subject=p.fs_subject_dir,
        subjects_dir=p.subj_brains_dir,
        hemi='both',
        views='dorsal',  # Top view
        time_viewer=False,
        colorbar=False,  # No built-in colorbar - we'll add our own
        clim=dict(kind='value', lims=[vmin, (vmin+vmax)/2, vmax]),
        colormap='RdBu_r',  # Same colormap as electrode topomap
        background='white',
        foreground='black',
        cortex='classic',
        size=(800, 800),
        smoothing_steps=20
    )
    
    # Add sensory cortex borders if available
    if s1_labels:
        for label in s1_labels:
            brain.add_label(label, borders=True, color='yellow', alpha=0.8)
    
    # Add hand disk labels if provided
    from mne.label import split_label

    # Add hand disk labels if provided
    os.environ['SUBJECTS_DIR'] = p.subj_brains_dir
    if hand_disk_labels:
        print(f" Adding hand disk labels to brain plot:")
        for hemi in ['lh', 'rh']:
            if hemi in hand_disk_labels and hand_disk_labels[hemi] is not None:
                label = hand_disk_labels[hemi]
                
                # Split into connected components
                sublabels = split_label(label)
                if len(sublabels) > 1:
                    # Choose the largest component
                    largest = max(sublabels, key=lambda x: len(x.vertices))
                    print(f"   {hemi.upper()} label split into {len(sublabels)} components, "
                          f"keeping largest with {len(largest.vertices)} vertices")
                    label = largest
                else:
                    print(f"   {hemi.upper()} label has {len(label.vertices)} vertices (single component)")
                
                brain.add_label(label, borders=True, color='black', alpha=1.0)

    
    # Save the brain image first
    filename = f'source_snr_average_{rep}reps.png'
    filepath = path.join(cwd, filename)
    brain.save_image(filepath)
    print(f" Saved brain image: {filepath}")
    
    brain.close()
    
    # Create a separate figure with custom colorbar
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.image as mpimg
    
    # Load the saved brain image
    brain_img = mpimg.imread(filepath)
    
    # Get image dimensions to preserve aspect ratio
    img_height, img_width = brain_img.shape[:2]
    aspect_ratio = img_width / img_height
    
    # Create figure with proper aspect ratio
    fig_width = 12
    fig_height = fig_width / aspect_ratio
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Main axes for brain image (preserve aspect ratio)
    ax_brain = plt.axes([0.05, 0.05, 0.80, 0.90])
    ax_brain.imshow(brain_img, aspect='equal')
    ax_brain.axis('off')
    
    # Colorbar axes on the right
    ax_cbar = plt.axes([0.88, 0.15, 0.03, 0.70])
    
    # Create colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, cax=ax_cbar)
    
    # === Font and line styling ===
    cbar.set_label('SNR', rotation=270, labelpad=25, fontsize=35, fontweight='bold')
    cbar.ax.tick_params(labelsize=30, width=1.8, length=6)  # thicker ticks, larger font
    
    # Make colorbar outline (spine) thicker and darker
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    # Optional: bold tick labels
    #for label in cbar.ax.get_yticklabels():
    #    label.set_fontweight('bold')

    
    # Save combined figure
    filename_with_cbar = f'source_snr_average_{rep}reps_with_colorbar.png'
    filepath_with_cbar = path.join(cwd, filename_with_cbar)
    plt.savefig(filepath_with_cbar, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f" Saved final image with colorbar: {filepath_with_cbar}")
    
    return vmin, vmax, avg_stc
#----------------------------------------------------------------------------------------#
def collect_hemisphere_electrode_data_from_averaged(averaged_electrode_dict, 
                                                    electrode_names, hemi,
                                                    geo_dists_scalp, electrode_positions):
    """
    Collect electrode data from pre-averaged SNR values (from topomaps).
    
    Parameters
    ----------
    averaged_electrode_dict : dict
        Dictionary with channel names as keys and averaged SNR as values
    electrode_names : list
        List of all electrode names
    hemi : str
        'lh' or 'rh'
    geo_dists_scalp : array
        Geodesic distances for electrodes
    electrode_positions : array
        3D positions of electrodes
        
    Returns
    -------
    electrode_avg : dict
        Dictionary with electrode info {name: {'snr': value, 'dist': distance, 'n': 1}}
    """
    print(f"\n[collect_hemisphere_electrode_data_from_averaged] Collecting averaged electrode data for {hemi.upper()}")
    
    electrode_avg = {}
    
    for i, elec_name in enumerate(electrode_names):
        if elec_name not in averaged_electrode_dict:
            continue
            
        # Get averaged SNR from the topomap data
        avg_snr = averaged_electrode_dict[elec_name]
        
        if np.isnan(avg_snr) or not np.isfinite(geo_dists_scalp[i]):
            continue
        
        electrode_avg[elec_name] = {
            'snr': avg_snr,
            'dist': geo_dists_scalp[i],
            'n': 1,  # n=1 since this is already averaged across subjects
            'pos': electrode_positions[i]
        }
    
    print(f" Collected {len(electrode_avg)} electrodes")
    print(f" SNR range: [{min(e['snr'] for e in electrode_avg.values()):.3f}, "
          f"{max(e['snr'] for e in electrode_avg.values()):.3f}]")
    
    return electrode_avg