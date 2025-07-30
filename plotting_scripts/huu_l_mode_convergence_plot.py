
## huu_l_mode_convergence_plot.py

from pybhpt.geo import KerrGeodesic
from scipy.special import ellipk
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import pandas as pd
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
plt.rcParams["font.size"] = 18

def full_libration(xp):
    return np.concatenate((xp, np.flip(xp[1:-1])))

def orbit_average_gen(func, geo, axis = 0):
    th = geo.polarpoints
    r = geo.radialpoints
    thp = full_libration(th)
    rp = full_libration(r)
    q = geo.blackholespin
    sig = np.add.outer(rp**2, q**2*np.cos(thp)**2)
    if func.shape[1] != 2:
        sigA = np.array(sig)
    else:
        sigA = np.array([sig, sig])
    return np.mean(np.mean(func*sigA, axis = axis), axis = axis-1)/np.mean(sig)

def huu_reg_gen_base(q, En, Lz, Qc, r, th):
    sig = r**2 + q**2*np.cos(th)**2
    eta = pow(sig,-1)*pow(2.*(2.*r + sig)*pow(q,2)*(sig*(-1.*Qc + pow(Lz,2)) + (2.*r + sig*pow(En,2))*(-1.*sig + pow(r,2))) + pow(q,4)*pow(2.*r + sig,2) + pow(sig*(Qc + pow(Lz,2)) - (2.*r + sig*pow(En,2))*(-1.*sig + pow(r,2)),2),0.5)
    zeta = Qc - 1.*r*(2. + r*(-2. + pow(En,2))) + sig*pow(En,2) + pow(Lz,2) + pow(q,2) + 2.*r*(pow(q,2) + pow(r,2))*pow(sig,-1)
    k = 2*eta/(eta + zeta)
    K = ellipk(k)
    return 4.*K/(np.pi*np.sqrt(eta/k))

def huu_reg_gen(geo):
    En = geo.orbitalenergy
    Lz = geo.orbitalangularmomentum
    Qc = geo.carterconstant
    r = geo.radialpoints
    r0 = full_libration(r)
    th = geo.polarpoints
    th0 = full_libration(th)
    q = geo.blackholespin
    rp = np.array([r0 for th in th0]).T
    thp = np.array([th0 for r in r0])
    return huu_reg_gen_base(q, En, Lz, Qc, rp, thp)

def load_huu_Yl_data(jobname, dir = None, param_dir = None):
    if dir is None:
        dir = os.path.dirname(os.path.realpath(__file__))
    if param_dir is None:
        param_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir, jobname)
    param_file = os.path.join(param_dir, f"params-{jobname}.json")
    with open(param_file) as json_file:
        params = json.load(json_file)
    lmax = params['lmax']
    nsamples = params['nsamples']
    meta_data = []
    print(lmax)

    huuYl_arr = np.zeros((2, lmax+1, nsamples, nsamples))
    for m in range(0, lmax + 1):
        try:
            data_file = os.path.join(data_dir, f"huuYl-{jobname}-km-m{m}.npy")
            data_lm = np.load(data_file).swapaxes(0,1).reshape(4, lmax+1, nsamples, nsamples)
            huuYl_arr += data_lm[:2]
        except FileNotFoundError:
            print(f"huuYl-{jobname}-km-m{m}.npy not found")
        try:
            data_file = os.path.join(data_dir, f"huuYl-{jobname}-kp-m{m}.npy")
            data_lm = np.load(data_file).swapaxes(0,1).reshape(4, lmax+1, nsamples, nsamples)
            huuYl_arr += data_lm[:2]
        except FileNotFoundError:
            print(f"huuYl-{jobname}-kp-m{m}.npy not found")
        try:
            data_file = os.path.join(data_dir, f"metadata-{jobname}-km-m{m}.txt")
            data_meta = np.loadtxt(data_file)
            meta_data.append(data_meta)
        except FileNotFoundError:
            print(f"metadata-{jobname}-km-m{m}.txt not found")
        try:
            data_file = os.path.join(data_dir, f"metadata-{jobname}-kp-m{m}.txt")
            data_meta = np.loadtxt(data_file)
            meta_data.append(data_meta)
        except FileNotFoundError:
            print(f"metadata-{jobname}-km-m{m}.txt not found")
    
    return huuYl_arr, params, meta_data

if __name__ == "__main__":
    data_dir = "../data"
    df = pd.read_csv(os.path.join(data_dir, "huu_metadata.csv"))
    df_highe = df[df['e'] == 0.6]

    huuYl_arr = np.load(os.path.join(data_dir, df_highe["filename_huu"].values[0]))
    with open(os.path.join(data_dir, df_highe["filename_params"].values[0])) as json_file:
        params = json.load(json_file)

    a = params['a']
    p = params['p']
    e = params['e']
    x = params['x']
    
    print(a,p,e,x)
    print(params["nsamples"])
    
    geo = KerrGeodesic(a, p, e, x, nsamples = params["nsamples"])

    fig, ax = plt.subplots(figsize=(6, 6))
    data0 = np.abs(orbit_average_gen(huuYl_arr[0] - huu_reg_gen(geo), geo, axis = 2))
    data1 = np.abs(orbit_average_gen(huuYl_arr[1] - huu_reg_gen(geo), geo, axis = 2))
    l0_list = np.arange(0., len(data0))
    lmax = l0_list[-1]
    lmax = 15
    ymin = np.min(np.abs(data0[:lmax]))/5
    ymax = 5*np.max(np.abs(data1[:lmax]))
    ax.plot(l0_list, data0, 'o', fillstyle='none', label="$r \\rightarrow r_p^-$")
    ax.plot(l0_list, data1, 'd', fillstyle='none', label="$r \\rightarrow r_p^+$")
    for i in range(25):
        ax.plot(l0_list[1:], ymax*10.**(3-0.5*i)*l0_list[1:]**(-2), '--', color='gray', lw=0.5)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels([1, 2, 5, 10, 20])
    ax.set_xlabel("$\\ell$")
    ax.set_ylabel("$\\left|\\langle h_{uu}^{\\mathrm{R}, \\ell} \\rangle\\right|$")
    # ax.set_title(f"$(a, p, e, x) = ({a}, {p:0.4f}, {e}, {x})$")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(1, lmax)
    ax.legend(loc='upper right', fontsize=12)
    plt.savefig(f"figures/huuYl-plot-{a}-{p:0.4f}-{e}-{x}.pdf", bbox_inches='tight')
