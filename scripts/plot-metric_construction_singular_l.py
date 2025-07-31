from pybhpt.geo import KerrGeodesic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "metric")
    df = pd.read_csv(os.path.join(data_dir, 'metric_metadata.csv'))

    a0 = 0.9
    p0 = 8.0
    lmin = 2
    lmax = 30

    df_subset = df.loc[(df['a'] == a0) & (df['p'] == p0) & (df['lmax'] == lmax)]

    gauges = df_subset['gauge'].unique()

    r_grid = np.load(os.path.join(data_dir, df_subset['rgrid_file'].values[0]))
    p0_loc = np.where(r_grid == p0)[0][0]
    lmax = df_subset['lmax'].values[0]
    spin, p0, e0, x0 = df_subset[['a', 'p', 'e', 'x']].values[0]

    kappa = np.sqrt(1-spin**2)
    rp = 1 + kappa
    rm = 1 - kappa

    hret_data_g = {}
    hret_l_g = {}
    for gauge, hret_file in zip(df_subset['gauge'], df_subset['hretlm_file']):
        hret_data_g[gauge] = np.load(os.path.join(data_dir, hret_file))
        hret_l_g[gauge] = np.sum(hret_data_g[gauge], axis=1)

    plt.rcParams["font.size"] = 12
    markers = ['o', 'x', 'd']
    fig, ax = plt.subplots(6, 3, figsize=(10.5, 12), sharex=True, sharey=True)
    lmodes = np.linspace(lmin, lmax + 1, lmax - lmin + 1)  
    
    ## Plotting IRG data in the first column
    labels = ["$\\left| h_{nn}^{\mathrm{IRG},\\ell} \\right|$", "$\\left| h_{n\\bar{m}}^{\mathrm{IRG},\\ell} \\right|$", "$\\left| h_{\\bar{m}\\bar{m}}^{\mathrm{IRG},\\ell} \\right|$"]
    for comp_iter_test in range(0, 3):
        data_inner = hret_l_g['IRG'][lmin:lmax + 1, comp_iter_test, 0, p0_loc]
        data_outer = hret_l_g['IRG'][lmin:lmax + 1, comp_iter_test, 1, p0_loc]
        if comp_iter_test == 0: # include complex conjugate
            data_inner = 2*np.real(data_inner)
            data_outer = 2*np.real(data_outer)
        ymax = 1.1*np.max([np.abs(data_inner[-1]), np.abs(data_outer[-1])])
        ax[comp_iter_test, 0].plot(lmodes, np.abs(data_inner), markers[0], ms = 3, fillstyle='none', label="$r \\rightarrow r_p^-$")
        ax[comp_iter_test, 0].plot(lmodes, np.abs(data_outer), markers[1], ms = 3, fillstyle='none', label="$r \\rightarrow r_p^+$")
        ax[comp_iter_test, 0].plot(lmodes, 0.5*np.abs(data_inner+data_outer), markers[2], ms = 3, fillstyle='none', label="average")
        ax[comp_iter_test, 0].plot(lmodes, ymax*(lmodes/lmax)**(0)/1.1, '--', color='gray', lw=0.5)
        ax[comp_iter_test, 0].set_yscale('log')
        ax[comp_iter_test, 0].set_ylabel(labels[comp_iter_test])

    ## Plotting ORG data in the second column
    labels = ["$\\left| h_{ll}^{\mathrm{ORG},\\ell} \\right|$", "$\\left| h_{lm}^{\mathrm{ORG},\\ell} \\right|$", "$\\left| h_{mm}^{\mathrm{ORG},\\ell} \\right|$"]
    for comp_iter_test in range(0, 3):
        data_inner = hret_l_g['ORG'][lmin:lmax + 1, comp_iter_test, 0, p0_loc]
        data_outer = hret_l_g['ORG'][lmin:lmax + 1, comp_iter_test, 1, p0_loc]
        if comp_iter_test == 0: # include complex conjugate
            data_inner = 2*np.real(data_inner)
            data_outer = 2*np.real(data_outer)
        ymax = 1.1*np.max([np.abs(data_inner[-1]), np.abs(data_outer[-1])])
        ax[3+comp_iter_test, 0].plot(lmodes, np.abs(data_inner), markers[0], ms = 3, fillstyle='none', label="$r \\rightarrow r_p^-$")
        ax[3+comp_iter_test, 0].plot(lmodes, np.abs(data_outer), markers[1], ms = 3, fillstyle='none', label="$r \\rightarrow r_p^+$")
        ax[3+comp_iter_test, 0].plot(lmodes, 0.5*np.abs(data_inner+data_outer), markers[2], ms = 3, fillstyle='none', label="average")
        ax[3+comp_iter_test, 0].plot(lmodes, ymax*(lmodes/lmax)**(0)/1.1, '--', color='gray', lw=0.5)
        ax[3+comp_iter_test, 0].set_yscale('log')
        ax[3+comp_iter_test, 0].set_ylabel(labels[comp_iter_test])

    ## Plotting SRG data in the third column
    #### First we account for SRG0 contributions
    labels = ["$\\left| h_{nn}^{\mathrm{SRG},\\ell} \\right|$", "$\\left| h_{n\\bar{m}}^{\mathrm{SRG},\\ell} \\right|$", "$\\left| h_{\\bar{m}\\bar{m}}^{\mathrm{SRG},\\ell} \\right|$"]
    for comp_iter_test in range(0, 3):
        data_inner = hret_l_g['SRG0'][lmin:lmax + 1, comp_iter_test, 0, p0_loc]
        data_outer = hret_l_g['SRG0'][lmin:lmax + 1, comp_iter_test, 1, p0_loc]
        if comp_iter_test == 0: # include complex conjugate
            data_inner = 2*np.real(data_inner)
            data_outer = 2*np.real(data_outer)
        if comp_iter_test == 2:
            data_inner += np.conj(hret_l_g['SRG4'][lmin:lmax + 1, comp_iter_test, 0, p0_loc])
            data_outer += np.conj(hret_l_g['SRG4'][lmin:lmax + 1, comp_iter_test, 1, p0_loc])
        ymax = 1.1*np.max([np.abs(data_inner[-1]), np.abs(data_outer[-1])])
        ax[comp_iter_test, 1].plot(lmodes, np.abs(data_inner), markers[0], ms = 3, fillstyle='none', label="$r \\rightarrow r_p^-$")
        ax[comp_iter_test, 1].plot(lmodes, np.abs(data_outer), markers[1], ms = 3, fillstyle='none', label="$r \\rightarrow r_p^+$")
        ax[comp_iter_test, 1].plot(lmodes, 0.5*np.abs(data_inner+data_outer), markers[2], ms = 3, fillstyle='none', label="average")
        ax[comp_iter_test, 1].plot(lmodes, ymax*(lmodes/lmax)**(0)/1.1, '--', color='gray', lw=0.5)
        ax[comp_iter_test, 1].set_yscale('log')
        ax[comp_iter_test, 1].set_ylabel(labels[comp_iter_test])

    #### Second we account for SRG4 contributions
    labels = ["$\\left| h_{ll}^{\mathrm{SRG},\\ell} \\right|$", "$\\left| h_{lm}^{\mathrm{SRG},\\ell} \\right|$", "$\\left| h_{mm}^{\mathrm{SRG},\\ell} \\right|$"]
    for comp_iter_test in range(0, 3):
        data_inner = hret_l_g['SRG4'][lmin:lmax + 1, comp_iter_test, 0, p0_loc]
        data_outer = hret_l_g['SRG4'][lmin:lmax + 1, comp_iter_test, 1, p0_loc]
        if comp_iter_test == 0: # include complex conjugate
            data_inner = 2*np.real(data_inner)
            data_outer = 2*np.real(data_outer)
        if comp_iter_test == 2:
            data_inner += np.conj(hret_l_g['SRG0'][lmin:lmax + 1, comp_iter_test, 0, p0_loc])
            data_outer += np.conj(hret_l_g['SRG0'][lmin:lmax + 1, comp_iter_test, 1, p0_loc])
        ymax = 1.1*np.max([np.abs(data_inner[-1]), np.abs(data_outer[-1])])
        ax[3+comp_iter_test, 1].plot(lmodes, np.abs(data_inner), markers[0], ms = 3, fillstyle='none', label="$r \\rightarrow r_p^-$")
        ax[3+comp_iter_test, 1].plot(lmodes, np.abs(data_outer), markers[1], ms = 3, fillstyle='none', label="$r \\rightarrow r_p^+$")
        ax[3+comp_iter_test, 1].plot(lmodes, 0.5*np.abs(data_inner+data_outer), markers[2], ms = 3, fillstyle='none', label="average")
        ax[3+comp_iter_test, 1].plot(lmodes, ymax*(lmodes/lmax)**(0)/1.1, '--', color='gray', lw=0.5)
        ax[3+comp_iter_test, 1].set_yscale('log')
        ax[3+comp_iter_test, 1].set_ylabel(labels[comp_iter_test])
    
    ## Plotting ARG data in the third column
    #### First we account for ARG0 contributions
    labels = ["$\\left| h_{nn}^{\mathrm{ARG},\\ell} \\right|$", "$\\left| h_{n\\bar{m}}^{\mathrm{ARG},\\ell} \\right|$", "$\\left| h_{\\bar{m}\\bar{m}}^{\mathrm{ARG},\\ell} \\right|$"]
    for comp_iter_test in range(0, 3):
        data_inner = -hret_l_g['ARG0'][lmin:lmax + 1, comp_iter_test, 0, p0_loc]
        data_outer = -hret_l_g['ARG0'][lmin:lmax + 1, comp_iter_test, 1, p0_loc]
        if comp_iter_test == 0: # include complex conjugate
            data_inner = 2*np.real(data_inner)
            data_outer = 2*np.real(data_outer)
        if comp_iter_test == 2:
            data_inner += np.conj(hret_l_g['ARG4'][lmin:lmax + 1, comp_iter_test, 0, p0_loc])
            data_outer += np.conj(hret_l_g['ARG4'][lmin:lmax + 1, comp_iter_test, 1, p0_loc])
        ymax = 1.1*np.max([np.abs(data_inner[-1]), np.abs(data_outer[-1])])
        ax[comp_iter_test, 2].plot(lmodes, np.abs(data_inner), markers[0], ms = 3, fillstyle='none', label="$r \\rightarrow r_p^-$")
        ax[comp_iter_test, 2].plot(lmodes, np.abs(data_outer), markers[1], ms = 3, fillstyle='none', label="$r \\rightarrow r_p^+$")
        ax[comp_iter_test, 2].plot(lmodes, 0.5*np.abs(data_inner+data_outer), markers[2], ms = 3, fillstyle='none', label="average")
        ax[comp_iter_test, 2].plot(lmodes, 0.5*np.abs(data_inner[-1]+data_outer[-1])*(lmodes/lmax)**(2), '--', color='gray', lw=0.5)
        ax[comp_iter_test, 2].plot(lmodes, ymax*(lmodes/lmax)**(3)/1.1, '--', color='gray', lw=0.5)
        ax[comp_iter_test, 2].set_yscale('log')
        ax[comp_iter_test, 2].set_ylabel(labels[comp_iter_test])
    
    #### Second we account for ARG4 contributions
    labels = ["$\\left| h_{ll}^{\mathrm{ARG},\\ell} \\right|$", "$\\left| h_{lm}^{\mathrm{ARG},\\ell} \\right|$", "$\\left| h_{mm}^{\mathrm{ARG},\\ell} \\right|$"]
    for comp_iter_test in range(0, 3):
        data_inner = hret_l_g['ARG4'][lmin:lmax + 1, comp_iter_test, 0, p0_loc]
        data_outer = hret_l_g['ARG4'][lmin:lmax + 1, comp_iter_test, 1, p0_loc]
        if comp_iter_test == 0: # include complex conjugate
            data_inner = 2*np.real(data_inner)
            data_outer = 2*np.real(data_outer)
        if comp_iter_test == 2:
            data_inner += -np.conj(hret_l_g['ARG0'][lmin:lmax + 1, comp_iter_test, 0, p0_loc])
            data_outer += -np.conj(hret_l_g['ARG0'][lmin:lmax + 1, comp_iter_test, 1, p0_loc])
        ymax = 1.1*np.max([np.abs(data_inner[-1]), np.abs(data_outer[-1])])
        ax[3+comp_iter_test, 2].plot(lmodes, np.abs(data_inner), markers[0], ms = 3, fillstyle='none', label="$r \\rightarrow r_p^-$")
        ax[3+comp_iter_test, 2].plot(lmodes, np.abs(data_outer), markers[1], ms = 3, fillstyle='none', label="$r \\rightarrow r_p^+$")
        ax[3+comp_iter_test, 2].plot(lmodes, 0.5*np.abs(data_inner+data_outer), markers[2], ms = 3, fillstyle='none', label="average")
        ax[3+comp_iter_test, 2].plot(lmodes, ymax*(lmodes/lmax)**(3)/1.1, '--', color='gray', lw=0.5)
        ax[3+comp_iter_test, 2].plot(lmodes, 0.5*np.abs(data_inner[-1]+data_outer[-1])*(lmodes/lmax)**(2), '--', color='gray', lw=0.5)
        ax[3+comp_iter_test, 2].set_yscale('log')
        ax[3+comp_iter_test, 2].set_xscale('log')
        ax[3+comp_iter_test, 2].set_xticks([2, 5, 10, 20, 35])
        ax[3+comp_iter_test, 2].set_xticklabels([2, 5, 10, 20, 35])
        ax[3+comp_iter_test, 2].set_ylabel(labels[comp_iter_test])

    ax[-1, 0].set_xlabel("$\\ell$")
    ax[-1, 1].set_xlabel("$\\ell$")
    ax[-1, 2].set_xlabel("$\\ell$")

    ax[0, 2].legend(loc='lower right', fontsize=10, ncol=2, frameon=True, reverse=False, alignment='right')
    plt.savefig(f"figures/hretlm_a{spin}_p{p0}_singular_lmode_convergence.pdf", bbox_inches='tight')