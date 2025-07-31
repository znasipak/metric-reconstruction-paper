# compute-z1-all.py
# This script computes the z1 values for all metrics in the dataset.

# compute-z1-from-load.py

from pybhpt.geo import KerrGeodesic
from pybhpt.geo import (kerrgeo_Vt_radial, 
                        kerrgeo_Vt_polar, 
                        kerrgeo_Vr, 
                        kerrgeo_Vtheta,
                        kerrgeo_Vphi_radial, 
                        kerrgeo_Vphi_polar)

from scipy.special import ellipk
import os
import json

import numpy as np

def full_libration(xp):
    return np.concatenate((xp, np.flip(xp[1:-1])))

def choose_spin_from_gauge(gauge):
    if gauge == "ORG" or gauge == "SAAB4" or gauge == "ASAAB4":
        s = 2
    else:
        s = -2
    return s

def huu_reg_gen_base(q, En, Lz, Qc, r, th):
    sig = r**2 + q**2*np.cos(th)**2
    eta = pow(sig,-1)*pow(2.*(2.*r + sig)*pow(q,2)*(sig*(-1.*Qc + pow(Lz,2)) + (2.*r + sig*pow(En,2))*(-1.*sig + pow(r,2))) + pow(q,4)*pow(2.*r + sig,2) + pow(sig*(Qc + pow(Lz,2)) - (2.*r + sig*pow(En,2))*(-1.*sig + pow(r,2)),2),0.5)
    zeta = Qc - 1.*r*(2. + r*(-2. + pow(En,2))) + sig*pow(En,2) + pow(Lz,2) + pow(q,2) + 2.*r*(pow(q,2) + pow(r,2))*pow(sig,-1)
    k = 2*eta/(eta + zeta)
    K = ellipk(k)
    return 4.*K/(np.pi*np.sqrt(eta/k))

def huu_reg_inc(geo):
    En = geo.orbitalenergy
    Lz = geo.orbitalangularmomentum
    Qc = geo.carterconstant
    r = geo.radialpoints[0]
    th = geo.polarpoints
    th0 = full_libration(th)
    q = geo.blackholespin
    return huu_reg_gen_base(q, En, Lz, Qc, r, th0)

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

def ut_gen(q, En, Lz, Qc, r, th):
    Sigma = np.add.outer(r**2, q**2*np.cos(th)**2)
    utr = np.array([kerrgeo_Vt_radial(q, En, Lz, Qc, ri) for ri in r])
    utth = np.array([kerrgeo_Vt_polar(q, En, Lz, Qc, thi) for thi in th])
    ut_rth = np.add.outer(utr, utth)
    return (ut_rth)/Sigma

def ur_gen(q, En, Lz, Qc, r, th):
    r0 = np.unique(r)
    urUp = np.sqrt(np.abs(np.array([kerrgeo_Vr(q, En, Lz, Qc, ri) for ri in r0])))
    urUp[0] = 0.
    urUp[-1] = 0.
    urDown = -np.flip(urUp)[1:-1]
    ur = np.concatenate((urUp, urDown))

    Sigma = np.add.outer(r**2, q**2*np.cos(th)**2)
    ur_rth = np.array([ur for thi in th]).T
    return ur_rth/Sigma

def uth_gen(q, En, Lz, Qc, r, th):
    th0 = np.unique(th)
    uthUp = np.sqrt(np.abs(np.array([kerrgeo_Vtheta(q, En, Lz, Qc, thi) for thi in th0])))
    uthUp[0] = 0.
    uthUp[-1] = 0.
    uthDown = -np.flip(uthUp)[1:-1]
    uth = np.concatenate((uthUp, uthDown))

    Sigma = np.add.outer(r**2, q**2*np.cos(th)**2)
    uth_rth = np.array([uth for ri in r])
    return uth_rth/Sigma

def uphi_gen(q, En, Lz, Qc, r, th):
    Sigma = np.add.outer(r**2, q**2*np.cos(th)**2)
    utr = np.array([kerrgeo_Vphi_radial(q, En, Lz, Qc, ri) for ri in r])
    utth = np.array([kerrgeo_Vphi_polar(q, En, Lz, Qc, thi) for thi in th])
    ut_rth = np.add.outer(utr, utth)
    return (ut_rth)/Sigma

def ut_gen_geo(geo):
    En = geo.orbitalenergy
    Lz = geo.orbitalangularmomentum
    Qc = geo.carterconstant
    r = full_libration(geo.radialpoints)
    th = full_libration(geo.polarpoints)
    q = geo.blackholespin
    return ut_gen(q, En, Lz, Qc, r, th)

def ur_gen_geo(geo):
    En = geo.orbitalenergy
    Lz = geo.orbitalangularmomentum
    Qc = geo.carterconstant
    r = full_libration(geo.radialpoints)
    th = full_libration(geo.polarpoints)
    q = geo.blackholespin
    return ur_gen(q, En, Lz, Qc, r, th)

def uth_gen_geo(geo):
    En = geo.orbitalenergy
    Lz = geo.orbitalangularmomentum
    Qc = geo.carterconstant
    r = full_libration(geo.radialpoints)
    th = full_libration(geo.polarpoints)
    q = geo.blackholespin
    return uth_gen(q, En, Lz, Qc, r, th)

def uphi_gen_geo(geo):
    En = geo.orbitalenergy
    Lz = geo.orbitalangularmomentum
    Qc = geo.carterconstant
    r = full_libration(geo.radialpoints)
    th = full_libration(geo.polarpoints)
    q = geo.blackholespin
    return uphi_gen(q, En, Lz, Qc, r, th)

def completion_gen(geo):
    En = geo.orbitalenergy
    Lz = geo.orbitalangularmomentum
    Qc = geo.carterconstant
    r = full_libration(geo.radialpoints)
    th = full_libration(geo.polarpoints)
    q = geo.blackholespin

    dM = En
    dJ = Lz
    z = np.cos(th)
    Sigma = np.add.outer(r**2, q**2*z**2)
    r2d = np.array([r for thi in th]).T
    z2d = np.array([z for ri in r])
    Delta = r2d**2 - 2*r2d + q**2

    htt_dM = 2 * r2d * ( r2d**2 + 3 * q**2 * z2d**2 ) / Sigma**2
    htt_dJ = -4 * q * r2d * z2d**2 / Sigma**2
    htt = (dM*htt_dM + dJ*htt_dJ)

    htphi_dM = -4 * r2d * q**3 * z2d**2 * ( 1. - z2d**2 ) / Sigma**2
    htphi_dJ = -2 * r2d * ( 1. - z2d**2 ) * ( r2d**2 - q**2 * z2d**2) / Sigma**2
    htphi = (dM*htphi_dM + dJ*htphi_dJ)

    hrr_dM = 2 * r2d * ( r2d**2 + 3 * q**2 + q**2 * ( r2d - 3 ) * ( 1. - z2d**2 ) ) / Delta**2
    hrr_dJ = -q * r2d * ( r2d + 2 + ( r2d - 2 ) * ( 1 - 2 * z2d**2 ) ) / Delta**2
    hrr = (dM*hrr_dM + dJ*hrr_dJ)

    hthth_dM = -2 * q**2 * z2d**2
    hthth_dJ = 2 * q * z2d**2
    hthth = (dM*hthth_dM + dJ*hthth_dJ)

    hphph_dM = -2 * q**2 * ( 1 - z2d**2 ) * ( Sigma**2 + r2d * ( r2d**2 - q**2 * z2d**2 ) * ( 1 - z2d**2 ) ) / Sigma**2
    hphph_dJ = 2 * q * ( 1 - z2d**2 ) * ( Sigma**2 + 2 * r2d**3 * ( 1 - z2d**2 ) ) / Sigma**2
    hphiphi = (dM*hphph_dM + dJ*hphph_dJ)

    ut = ut_gen(q, En, Lz, Qc, r, th)
    uphi = uphi_gen(q, En, Lz, Qc, r, th)
    ur = ur_gen(q, En, Lz, Qc, r, th)
    uth = uth_gen(q, En, Lz, Qc, r, th)

    return (htt*ut*ut + 2*htphi*ut*uphi + hrr*ur*ur + hthth*uth*uth + hphiphi*uphi*uphi)

def z0_gen_geo(geo):
    En = geo.orbitalenergy
    Lz = geo.orbitalangularmomentum
    Qc = geo.carterconstant
    r = full_libration(geo.radialpoints)
    th = full_libration(geo.polarpoints)
    q = geo.blackholespin

    Sigma = np.add.outer(r**2, q**2*np.cos(th)**2)
    return np.mean(Sigma)/np.mean(ut_gen(q, En, Lz, Qc, r, th)*Sigma)

from scipy.optimize import lsq_linear
def lbasis(lmax, nmin = 0, nmax = 2):
    reg_sum = [1.]
    # reg_list = [((lmax + 1)), ((lmax + 1)/(2*lmax + 1)/(2*lmax + 3)), ((lmax + 1)/(2*lmax + 1)/(2*lmax + 3)/(2*lmax - 1)/(2*lmax + 5))/3, ((lmax + 1)/(2*lmax + 1)/(2*lmax + 3)/(2*lmax - 1)/(2*lmax + 5)/(2*lmax - 3)/(2*lmax + 7))/5, ((lmax + 1)/(2*lmax + 1)/(2*lmax + 3)/(2*lmax - 1)/(2*lmax + 5)/(2*lmax - 3)/(2*lmax + 7)/(2*lmax - 3)/(2*lmax + 7))/5]
    reg_list = [lmax+1]
    for n in range(1, nmax + 1):
        reg_param = (lmax + 1)/(2*(n - 1) + 1)
        for i in range(1, n+1):
            reg_param *= 1/(2*lmax + 1 + 2*i)/(2*lmax + 3 - 2*i)
        reg_list.append(reg_param)
    if nmax > len(reg_list) - 1:
        nmax = len(reg_list) - 1
    return np.concatenate([reg_sum, reg_list[nmin:nmax+1]])

def lweight(l, n):
    weight = 1
    for i in range(1, n + 1):
        weight *= 1/(2*l + 1 + 2*i)/(2*l + 1 - 2*i)
    return weight

def lfit(huul, lminTest, lmaxTest, nmin = 0, nmax = 2):
    ysum = np.array([np.sum(huul[:lmax + 1]) for lmax in range(lminTest, lmaxTest + 1)])
    xsum = np.array([lbasis(lmax, nmin = nmin, nmax = nmax) for lmax in range(lminTest, lmaxTest + 1)])
    b = lsq_linear(xsum, ysum)
    return b.x

def huu_fit(huuYl, huuReg, geo, nmin = 1, nmax = 2, min_lmax = 12, lmin_test = 8, axis = 3):
    """
    Extrapolate high-$l$ behaviour of huu based on singular structure.
    """

    # If given inner and outer most limits (dim 0 == 2) then take the average of the two
    mean_flag = False
    if huuYl.shape[0] == 2:
        lmax = huuYl.shape[1] - 1
        mean_flag = True
    else:
        lmax = huuYl.shape[0] - 1

    # Now remove the leading-order singular structure and orbit average, e.g., get $\langle z_1 \rangle_t$
    datareg_avg = orbit_average_gen(huuYl - huuReg, geo, axis = axis)
    if mean_flag:
        datareg_avg = np.mean(datareg_avg, axis = 0)
    
    # Now we fit for the data, but vary the number of modes we include in our fit, since higher $l$-modes
    # can contain more noise and spoil the fit
    huu_fits = []
    for lmax_test in range(min_lmax, lmax):
        # Each fitting procedure gives an array of estimates
        b = np.array([lfit(datareg_avg, lmin_var, lmax_test, nmin = nmin, nmax = nmax) for lmin_var in range(lmin_test, lmax_test)])
        
        # We take the mean of the fit coefficients, and the standard deviation as the error
        huu_fits.append([np.mean(b[:, 0], axis = 0), 2*np.std(b[:, 0], axis = 0), lmax_test, *np.mean(b[:, 1:], axis = 0)])
    huu_fits = np.array(huu_fits)
    
    # Now we need to add the completion term
    huu_comp = orbit_average_gen(completion_gen(geo), geo)
    
    # Finally, we find the best fit, and compute the z1 value
    best_fit = huu_fits[np.argmin(huu_fits[:, 1], axis = 0)]
    huu_recon = best_fit[0]
    huu_err = best_fit[1]

    # lmax_test is the lmax of the best fit, and we use it to compute the z1 value
    lmax_test = int(np.abs(best_fit[2]))
    
    # This estimates the higher order regularization coefficients Hn
    Hn_fits = best_fit[3:]
    
    # This is the contribution from the lmax_test mode
    huu_lmax = datareg_avg[lmax_test]
    
    # We then use standard quadrature error propagation to compute the z1 error
    for n in range(nmin, nmax):
        huu_lmax *= Hn_fits[n - nmin]*lweight(lmax_test, n)
    
    huu_val = huu_recon + huu_comp
    z0 = z0_gen_geo(geo)
    z1 = -0.5*z0*huu_val
    z1_err = 0.5*z0*huu_err

    # We then repeat this analysis for the second best fit, which is the one with the second lowest error
    huu_fits_2 = huu_fits.copy()
    huu_fits_2[np.argmin(huu_fits[:, 1], axis = 0), 1] = 1e6
    best_fit_2 = huu_fits_2[np.argmin(huu_fits_2[:, 1], axis = 0)]
    huu_recon_2 = best_fit_2[0]
    huu_err_2 = best_fit_2[1]
    lmax_test_2 = int(np.abs(best_fit_2[2]))
    Hn_fits_2 = best_fit_2[3:]
    huu_lmax_2 = datareg_avg[lmax_test_2]
    for n in range(nmin, nmax):
        huu_lmax_2 *= Hn_fits_2[n - nmin]*lweight(lmax_test_2, n)
    huu_val_2 = huu_recon_2 + huu_comp
    z1_2 = -0.5*z0*huu_val_2
    z1_err_2 = 0.5*z0*huu_err_2

    # Return the results in a dictionary
    # The dictionary contains the z0 value, the huu fits, the completion term, and the z1 estimates
    # The z1 estimates are in the form of a dictionary with the z1 value, the error, the lmax of the fit, and the contribution from the lmnx_test mode
    return {"z0": z0, 
            "fits": huu_fits,
            "completion": huu_comp,
            "est1": {"z1": z1, "err": z1_err, "lmax": lmax_test, "z1_lmax": -0.5*z0*huu_lmax},
            "est2": {"z1": z1_2, "err": z1_err_2, "lmax": lmax_test_2, "z1_lmax": -0.5*z0*huu_lmax_2},
            }

import pandas as pd
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "huu")
    df_full = pd.read_csv(os.path.join(data_dir, "huu_metadata.csv"))

    subdir = df_full['subdir'].unique()
    
    for s in subdir:
        df = df_full.loc[df_full['subdir'] == s].reset_index(drop=True)
        print("==========================")
        print("==========================")
        print(f"Processing subdir: {s}")
        print("==========================")
        print("==========================")
        print(df)

        data_list = []
        z1_str_list = []
        paper_iter_list = []
        max_iter = len(df)

        for i in range(max_iter):
            print(i)
            gauge = df.loc[i, 'gauge']
            name = df.loc[i, 'name']

            print(f"Processing {gauge} data for {name}...")
            print("==========================")

            params_file = os.path.join(data_dir, s, df.loc[i, 'params_file'])

            with open(params_file) as json_file:
                params = json.load(json_file)
            huuYl_file = os.path.join(data_dir, s, df.loc[i, 'huu_file'])
            huuYl_arr = np.load(huuYl_file)

            a = params['a']
            p = params['p']
            e = params['e']
            x = params['x']

            print(f"Computing z1 for {gauge} with parameters: ({a}, {p}, {e}, {x})")
            geo = KerrGeodesic(a, p, e, x, nsamples = params["nsamples"])
            
            huuYl_test = np.mean(huuYl_arr, axis = 0)

            if e == 0. and x**2 < 1.:
                huuYl_test = huuYl_test[:, np.newaxis, :]
                huureg = huu_reg_inc(geo)[np.newaxis, :]
            else:
                huureg = huu_reg_gen(geo)

            print("Fitting data...")
            out = huu_fit(huuYl_test, huureg, geo, lmin_test = 5, min_lmax = 12, axis = 2, nmax = 3)
            z1 = out["est1"]["z1"]
            z1_err = out["est1"]["err"]
            z1_lmax = out["est1"]["z1_lmax"]
            z0 = out["z0"]
            lmax_cut = out["est1"]["lmax"]
            data = [gauge, name, a, p, e, x, z0, z1, z1_err, lmax_cut, z1_lmax]
            data_list.append(data)

            error_order = int(-np.floor(np.log10(z1_err)))
            error_digit = int(np.ceil(z1_err*10**(error_order)))
            if error_digit == 10:
                error_digit = 1
                error_order -= 1
            z1_trunc = int(np.round(z1*10**error_order))/10**error_order

            print("---------------------")
            print(f"{z1_trunc}({error_digit})")
            print(f"Computed value z1: {z1}")
            print(f"Computed value z1 error: {z1_err}")
            print(f"Computed value z1 lmax: {z1_lmax}")

            # Save the results
            output_dir = f"results/{gauge}/{name}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, f"z1-{gauge}-{name}.npy")
            np.save(output_file, data)
            print(f"Results saved to {output_file}")

            z1_str_list.append(f"{z1_trunc}({error_digit})")
            paper_iter_list.append(i)

        df_data = pd.DataFrame(data_list, columns=['gauge', 'name', 'a', 'p', 'e', 'x', 'z0', 'z1', 'z1_err', 'lmax_cut', 'z1_lmax'])
        df_data.sort_values(by=['a', 'e', 'name', 'gauge'], inplace=True, ignore_index=True)
        df_data.to_csv(
            f"results/z1-{s}.csv"
        )