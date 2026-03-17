# -*- coding: utf-8 -*-
import mne
from os import getcwd
from os import path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d


def convert_to_erp(p, epoch_duration=None):
    print('Converting epochs to erp: started')
    cwd = getcwd()
    processedeegpath = path.join(cwd, p.data, p.processed_data, 
                                  p.protocol, p.subj, p.session, 
                                  p.eeg_dir)
    
    fullpath_epochs = path.join(processedeegpath,p.epochsfile)
    epochs = mne.read_epochs(fullpath_epochs)

    # for subj05
    #epochs.info['bads'] += ['PO5', 'PPO9H', 'PO7','TPP7h', 'PPO5h', 'T7', 'TTP7h', 'P7', 'TP7', 'PO9', 'CPP5h', 'TPP9h', 'CP5', 'P9','P3'] 

    # for subj06
    #epochs.info['bads'] += [ 'TPP7h']

    #for subj07
    #epochs.info['bads'] += [ 'F5', 'F7']

    erp = epochs[p.selected_epoch].average()
    erp.plot()

   # Define 4 different groups of epochs
    np.random.seed(11)
    
    group_sizes = [50, 100, 500, 2000]
    group_epochs = []
    
    n_epochs = len(epochs)
    
    for size in group_sizes:
        if size > n_epochs:
            raise ValueError("Group size exceeds the total number of epochs.")
    
        sampled_indices = np.random.choice(n_epochs, size=size, replace=False)
        sampled_epochs = epochs[sampled_indices]
        group_epochs.append(sampled_epochs.average())

    
    # Updated colors (colorblind-friendly greyscale)
    channel_colors = {"CP3": "#424242", "CPz": "#5c5b59", "CP4": "#999896"}   #channel_colors = {"CP3": "#424242", "CP4": "#999896"} 
    plot_channels = ["CP3", "CPz", "CP4"]   # plot_channels = ["CP3", "CP4"] 
    
    time_to_plot = 0.05  # Single time point for topomap
    
    # Step 1: Determine global min/max amplitude
    global_ymin, global_ymax = float("inf"), float("-inf")
    for erp in group_epochs:
        evoked = erp.copy().pick(plot_channels)
        data = evoked.data * 1e6
        global_ymin = min(global_ymin, data.min())
        global_ymax = max(global_ymax, data.max())
    
    
    plt.rcParams.update({
        'font.size': 20,        # Default text size
        'axes.labelsize': 20,   # Axis labels
        'axes.titlesize': 20,   # Subplot titles
        'xtick.labelsize': 20,  # X-axis tick labels
        'ytick.labelsize': 20,  # Y-axis tick labels
        'legend.fontsize': 18,  # Legend text
    })
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # --- Plot ERP Line Plots (Top Row) ---
    for i, erp in enumerate(group_epochs):
        ax = axes[0, i]  # Top row
        evoked_line = erp.copy().pick(plot_channels)
        
        times = evoked_line.times
        data = evoked_line.data * 1e6  # Convert to microvolts
    
        # Publication-friendly greyscale palette
        channel_colors = {"CP3": "#1a1a1a", "CPz": "#595959", "CP4": "#b3b3b3"}  # dark to light grey,  channel_colors = {"CP3": "#1a1a1a", "CP4": "#b3b3b3"} 
        line_styles = ['-', '--', '-.']  # solid, dashed, dash-dot line_styles = ['-', '--'] 
        line_width = 2.5  # thicker for print visibility
    
        for j, ch_name in enumerate(plot_channels):
            ch_idx = evoked_line.ch_names.index(ch_name)
            ax.plot(
                times,
                data[ch_idx],
                color=channel_colors[ch_name],
                linestyle=line_styles[j],
                linewidth=line_width,
                label=ch_name
            )
    
        # Add reference lines
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    
        # Highlighted 50 ms line
        ax.axvline(time_to_plot, color="black", linestyle="-", linewidth=2.0)
        
        ax.text(
            time_to_plot + 0.005, global_ymax * 0.9,
            "t = 50 ms", fontsize=16, fontweight="bold",
            verticalalignment="top", horizontalalignment="left"
        )
    
        # Global axis settings
        ax.set_ylim(global_ymin, global_ymax)
        ax.set_xlabel("Time (s)")
        ax.set_title("")
        if i == 0:
            ax.set_ylabel("Amplitude (µV)")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', length=0)
        ax.tick_params(axis="both", width=1.2)
        ax.grid(False)
    
        # Legend only for rightmost plot
        if i == 3:
            ax.legend(
                loc='lower right',
                frameon=False,
                handlelength=2.5,
                fontsize=18
            )
            
        # Add N_ave text
        n_ave = evoked_line.nave
        ax.text(0.98, 0.02, f'N_ave = {n_ave}', transform=ax.transAxes,
                fontsize=14, verticalalignment='bottom', horizontalalignment='right')
        
        # Add legend only to the rightmost subplot (i == 3)
        if i == 3:
            ax.legend(loc='lower right', frameon=True)

    # --- Topomap Settings ---
    vmin, vmax = -3, 3
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    time_idx = np.argmin(np.abs(erp.times - time_to_plot))
    
    for i, erp in enumerate(group_epochs):
        ax = axes[1, i]  # Bottom row
    
        # Add colorbar axis
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
    
        # Plot topomap
        erp.plot_topomap(
            times=[time_to_plot],
            sensors=True,
            contours=15,
            show_names=False,
            time_unit='s',
            cmap="coolwarm",
            axes=ax,
            colorbar=False,
            show=False
        )
        ax.set_title("") 
    
        # Colorbar
        mappable = ax.images[0]
        cb = fig.colorbar(mappable, cax=cax)
        
        # Add µV label to the last (rightmost) colorbar
        if i == 0:
            #cax.set_ylabel("µV", fontsize=18, labelpad=10)
            #cax.yaxis.set_label_position("right")
            # Optional: Add text below the bar instead (if preferred style)
            cax.text(0.5, -0.05, "µV", transform=cax.transAxes,
                     ha='center', va='top', fontsize=18)

            ax.text(
                0.5, -0.15,  # x,y in axes coordinates (0–1 range)
                "t = 50 ms",
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=18
            )
    
        # Overlay selected electrode names
        highlight_channels = ["CP3", "CPz", "CP4"]
        
        from mne.viz.topomap import _prepare_topomap_plot
        data = erp.data[:, erp.time_as_index(time_to_plot)[0]]
        out = _prepare_topomap_plot(erp.info, ch_type='eeg')
        pos = out[1]
        
        name_to_xy = dict(zip(erp.ch_names, pos))
        
        for ch in highlight_channels:
            if ch in name_to_xy:
                x, y = name_to_xy[ch]
                ax.plot(
                    x, y, 'o', color=channel_colors.get(ch, "gray"),
                    markersize=10, markeredgecolor="black", markeredgewidth=1.5
                )
                if i == 0:
                    ax.text(
                        x, y - 0.035, ch,
                        fontsize=18, ha='center', va='bottom', color='black'
                    )
    
    plt.tight_layout()
    plt.show()

    print('Converting epochs to erp: completed')