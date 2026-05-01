from pybhpt.geo import KerrGeodesic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib as mpl

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def add_power_law_guides(axis, left_power=-1, right_power=1, left_scale=1, right_scale=1, left_offset=2, right_offset=100):
    def region_max(xmin, xmax):
        max_y = 0.0
        for line in axis.lines:
            xdata = np.asarray(line.get_xdata())
            ydata = np.asarray(line.get_ydata())
            if xdata.size == 0 or ydata.size == 0:
                continue
            mask = (xdata >= xmin) & (xdata <= xmax) & np.isfinite(xdata) & np.isfinite(ydata) & (ydata > 0)
            if np.any(mask):
                max_y = max(max_y, np.max(ydata[mask]))
        return max_y
    
    def region_mean(xmin, xmax):
        mean_y = 0.0
        count = 0
        for line in axis.lines:
            xdata = np.asarray(line.get_xdata())
            ydata = np.asarray(line.get_ydata())
            if xdata.size == 0 or ydata.size == 0:
                continue
            mask = (xdata >= xmin) & (xdata <= xmax) & np.isfinite(xdata) & np.isfinite(ydata) & (ydata > 0)
            if np.any(mask):
                mean_y += np.sum(ydata[mask])
                count += np.sum(mask)
        return mean_y / count if count > 0 else 0.0

    x_left = np.logspace(np.log10(1e-3), np.log10(1e-1), 200)
    x_right = np.logspace(np.log10(5e1), np.log10(5e3), 200)

    left_max = region_mean(x_left[0], x_left[-1])
    right_max = region_mean(x_right[0], x_right[-1])

    if left_max <= 0:
        left_max = axis.get_ylim()[1]
    if left_max >= 1e8:
        left_max = 1e8
    if right_max <= 0:
        right_max = axis.get_ylim()[1]
    if right_max >= 1e8:
        right_max = 1e8

    left_anchor_y = left_max * left_scale
    left_scale = left_anchor_y / x_left[0]**left_power
    left_y = left_scale * x_left**left_power
    axis.plot(x_left, left_y, 'k--', lw=2, alpha=0.9, zorder=0)
    left_text_x = np.sqrt(x_left[0] * x_left[-1])
    left_text_y = left_scale * left_text_x**left_power * 1.25
    # rotate label to match the visual slope of the dashed line
    try:
        trans = axis.transData
        i0 = len(x_left) // 4
        i1 = 3 * len(x_left) // 4
        p0 = trans.transform((x_left[i0], left_y[i0]))
        p1 = trans.transform((x_left[i1], left_y[i1]))
        angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    except Exception:
        angle = 0.0
    axis.text(left_text_x, left_text_y * left_offset, r'$\sim \Delta^{' + str(left_power) + r'}$', ha='center', va='bottom', fontsize=13, color='black', rotation=angle, rotation_mode='anchor')

    right_anchor_y = right_max * right_scale
    right_scale = right_anchor_y / x_right[-1]**right_power
    right_y = right_scale * x_right**right_power
    axis.plot(x_right, right_y, 'k--', lw=2, alpha=0.9, zorder=0)
    right_text_x = np.sqrt(x_right[0] * x_right[-1])
    right_text_y = right_scale * right_text_x**right_power * 0.72
    # rotate label to match the visual slope of the dashed line
    try:
        trans = axis.transData
        j0 = len(x_right) // 4
        j1 = 3 * len(x_right) // 4
        q0 = trans.transform((x_right[j0], right_y[j0]))
        q1 = trans.transform((x_right[j1], right_y[j1]))
        angle_r = np.degrees(np.arctan2(q1[1] - q0[1], q1[0] - q0[0]))
    except Exception:
        angle_r = 0.0
    axis.text(right_text_x, right_text_y * right_offset, r'$\sim r^{' + str(right_power) + r'}$', ha='center', va='top', fontsize=13, color='black', rotation=angle_r, rotation_mode='anchor')

if __name__ == "__main__":
    # Load metadata file which tells us which data files to load
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "metric")
    df = pd.read_csv(os.path.join(data_dir, 'metric_metadata.csv'))

    # Select a specific system to plot
    a0 = 0.9
    ptemp = 8.0
    lmin = 2
    lmax = 30

    savepath = os.path.join(os.path.dirname(__file__), "..", "figures", f"hret_asymp_a{a0}_p{ptemp}_reduced.pdf")

    left_power = [-2, 0, 2, -1, 1, 0]
    left_scale = [1e-2, 0, 1e0, 5e2, 5e0, 5e-3]
    left_offset = [1e-3, 1, 1e1, 1e1, 1e1, 3e-3]
    right_power = [-3, 0, 1, -2, 1, -1]
    right_scale = [1e-2, 0, 5e-2, 5e-7, 5e-2, 1e-4]
    right_offset = [1000, 0, 0.1, 0.1, 0.1, 0.1]

    # Filter the DataFrame for the specific parameters
    df_subset = df.loc[(df['a'] == a0) & (df['p'] == ptemp) & (df['lmax'] == lmax)]
    gauges = df_subset['gauge'].unique()

    # Load the radial grid from the first file in the subset
    r_grid = np.load(os.path.join(data_dir, df_subset['rgrid_file'].values[0]))
    p0_loc = np.where(r_grid == ptemp)[0][0]
    lmax_temp = df_subset['lmax'].values[0]
    assert lmax_temp == lmax, f"Expected lmax {lmax}, but got {lmax_temp}"
    spin, p0, e0, x0 = df_subset[['a', 'p', 'e', 'x']].values[0]
    assert spin == a0, f"Expected spin {a0}, but got {spin}"
    assert p0 == ptemp, f"Expected p {ptemp}, but got {p0}"

    # Initialize KerrGeodesic object
    geo = KerrGeodesic(spin, p0, e0, x0, nsamples = 4)

    kappa = np.sqrt(1-spin**2)
    rp = 1 + kappa
    rm = 1 - kappa

    # Load hret data for each gauge and sum over l-modes
    hret_data_g = {}
    hret_m_g = {}
    for gauge, hret_file in zip(df_subset['gauge'], df_subset['hretlm_file']):
        hret_data_g[gauge] = np.load(os.path.join(data_dir, hret_file))
        hret_m_g[gauge] = np.sum(hret_data_g[gauge], axis=0)

    # Generate special time slices that are used to more nicely plot the asymptotic behavior
    Rstar_grid = r_grid + (1+kappa)/kappa*np.log((r_grid - rp)/2) - (1-kappa)/kappa*np.log((r_grid - rm)/2)
    Phistar_grid = spin / (2*kappa) * np.log((r_grid - rm)/(r_grid - rp))
    t_of_u_grid = 0 - Rstar_grid # u = t - Rstar
    t_of_u_grid[r_grid==p0] = 0 # this is to maintain t-slicing at r0
    t_of_v_grid = 0 + Rstar_grid # v = t + Rstar
    t_of_v_grid[r_grid==p0] = 0 # this is to maintain t-slicing at r0
    phi_grid = 0 + Phistar_grid # phi tilde = phi - Phistar

    # Prepare all of the components of the metric perturbation data by multiplying m-modes by coordinates above
    hret_inner = {}
    hret_outer = {}
    gauge_iter_dict = {'ORG': [0, 1, 2, 6, 7], # indices refer to ['ll', 'lm', 'lmbar', 'nn', 'nm', 'nmbar', 'mm', 'mbarmbar']
                       'SRG4': [0, 1, 2, 6, 7], # note that 'ln' and 'mmbar' are zero due to the gauge conditions (i.e., trace-free)
                       'ARG4': [0, 1, 2, 6, 7],
                       'IRG': [3, 5, 4, 7, 6],
                       'SRG0': [3, 5, 4, 7, 6],
                       'ARG0': [3, 5, 4, 7, 6]}
    for gauge in gauges:
        data_inner = {}
        data_outer = {}
        for i in range(3):
            data_inner[i] = np.zeros(r_grid.shape, dtype=np.complex128)
            data_outer[i] = np.zeros(r_grid.shape, dtype=np.complex128)
            for m in range(-lmax, lmax + 1):
                data_inner[i]+=hret_m_g[gauge][m,i,0]*np.exp(-1.j*geo.mode_frequency(m, 0, 0)*t_of_u_grid)*np.exp(1.j*m*phi_grid)
                data_outer[i]+=hret_m_g[gauge][m,i,1]*np.exp(-1.j*geo.mode_frequency(m, 0, 0)*t_of_v_grid)*np.exp(1.j*m*phi_grid)
            data_inner[i][data_inner[i] == 0] = None
            data_outer[i][data_outer[i] == 0] = None
        data_inner_row = [data_inner[0].copy(), data_inner[1].copy(), np.conj(data_inner[1].copy()), data_inner[2].copy(), np.conj(data_inner[2].copy())]
        data_outer_row = [data_outer[0].copy(), data_outer[1].copy(), np.conj(data_outer[1].copy()), data_outer[2].copy(), np.conj(data_outer[2].copy())]
        if gauge in gauge_iter_dict:
            gauge_iter = gauge_iter_dict[gauge]
            hret_inner[gauge] = np.zeros((8,) + data_inner_row[0].shape, dtype=np.complex128)
            hret_inner[gauge][gauge_iter] = data_inner_row
            hret_outer[gauge] = np.zeros((8,) + data_outer_row[0].shape, dtype=np.complex128)
            hret_outer[gauge][gauge_iter] = data_outer_row


    # Combine the two potentials for the SRG and ARG gauges
    hret_inner["SRG"] = [hret_inner["SRG4"][i] + hret_inner["SRG0"][i] for i in range(len(hret_inner["SRG0"]))]
    hret_outer["SRG"] = [hret_outer["SRG4"][i] + hret_outer["SRG0"][i] for i in range(len(hret_outer["SRG0"]))]
    hret_inner["ARG"] = [hret_inner["ARG4"][i] - hret_inner["ARG0"][i] for i in range(len(hret_inner["ARG0"]))]
    hret_outer["ARG"] = [hret_outer["ARG4"][i] - hret_outer["ARG0"][i] for i in range(len(hret_outer["ARG0"]))] 

    # Plot the results
    gauge_list = ['ORG', 'IRG', 'SRG', 'ARG']
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(3, 2, figsize=(6.5, 9), sharex=True, sharey=True)
    fig.tight_layout(pad=1.0)
    ax = ax.flatten()
    comp_list = ['$|h_{ll}^\\mathrm{G,\\pm}|$',
                '$|\\mathrm{Re}[h_{lm}^\\mathrm{G,\\pm}]|$',
                '$|\\mathrm{Im}[h_{l\\bar{m}}^\\mathrm{G,\\pm}]|$',
                '$|h_{nn}^\\mathrm{G,\\pm}|$',
                '$|\\mathrm{Re}[h_{nm}^\\mathrm{G,\\pm}]|$',
                '$|\\mathrm{Im}[h_{n\\bar{m}}^\\mathrm{G,\\pm}]|$',
                '$|\\mathrm{Re}[h_{mm}^\\mathrm{G,\\pm}]|$',
                '$|\\mathrm{Im}[h_{\\bar{m}\\bar{m}}^\\mathrm{G,\\pm}]|$',
                ]
    
    # comp_list = ['$|ll|$',
    #             '$|\\mathrm{Re}[lm]|$',
    #             '$|\\mathrm{Im}[l\\bar{m}]|$',
    #             '$|nn|$',
    #             '$|\\mathrm{Re}[nm]|$',
    #             '$|\\mathrm{Im}[n\\bar{m}]|$',
    #             '$|\\mathrm{Re}[mm]|$',
    #             '$|\\mathrm{Im}[\\bar{m}\\bar{m}]|$',
    #             ]
    line_list = ['-', '--', '-.', ':']
    hret_inner_list = [hret_inner[gauge] for gauge in gauge_list]
    hret_outer_list = [hret_outer[gauge] for gauge in gauge_list]
    color_list = mpl.cm.viridis(np.linspace(0, 0.8, len(hret_inner_list)))
    skip_components = [1,4,7]  # Skip plotting h_nn and h_nm components for reduced figure
    axis_dict = {0: 0, 1: 20, 2: 3, 3: 2, 4: 20, 5: 4, 6: 5, 7: 20}  # Map component index to axis index for skipped components

    for k in range(7):
        for j in range(4):
            data_inner = hret_inner_list[j][k]
            data_outer = hret_outer_list[j][k]
            if k == 1:
                data_inner = data_inner.real
                data_outer = data_outer.real
            elif k == 2:
                data_inner = data_inner.imag
                data_outer = data_outer.imag
            elif k == 4:
                data_inner = data_inner.real
                data_outer = data_outer.real
            elif k == 5:
                data_inner = data_inner.imag
                data_outer = data_outer.imag
            elif k == 6:
                data_inner = data_inner.real
                data_outer = data_outer.real
            elif k == 7:
                data_inner = data_inner.imag
                data_outer = data_outer.imag
            kk = axis_dict[k]

            if k in skip_components:
                continue
            if data_inner is not None and np.all(data_inner != 0.):
                ax[kk].plot(r_grid - rp, np.abs(data_inner), line_list[j], c=color_list[j], label = gauge_list[j])
            if data_outer is not None and np.all(data_outer != 0.):
                ax[kk].plot(r_grid - rp, np.abs(data_outer), line_list[j], c=color_list[j])
        if k in skip_components:
            continue
        ax[kk].plot((p0 - rp, p0 - rp), (1e-11, 1e8), 'k-', alpha = 0.2)
        ax[kk].set_ylabel(comp_list[k])
        ax[kk].set_yscale('log')
        ax[kk].set_xscale('log')
        ax[kk].set_xlim(1e-4, 2e4)
        ax[kk].set_ylim(1e-11, 1e8)
        add_power_law_guides(ax[kk], 
                             left_power=left_power[kk], 
                             right_power=right_power[kk], 
                             left_scale=left_scale[kk], 
                             right_scale=right_scale[kk],
                             left_offset=left_offset[kk],
                             right_offset=right_offset[kk]
                             )
    
    ax[kk].set_xlabel('$(r - r_+)/M$')
    ax[kk-1].set_xlabel('$(r - r_+)/M$')
    ax[kk].legend(loc='upper left', fontsize=12)
    ax[1].set_visible(False)
    print(f"Saving figure for spin {spin} and p0 {p0}")
    plt.savefig(savepath, dpi=300, bbox_inches='tight')