# compute-z1-all.py
# This script computes the z1 values for all metrics in the dataset.

# compute-z1-from-load.py

from pybhpt.geo import KerrGeodesic
from pybhpt.teuk import TeukolskyMode
from pybhpt.hertz import HertzMode
from pybhpt.redshift import RedshiftCoefficients
from pybhpt.geo import (kerrgeo_Vt_radial, 
                        kerrgeo_Vt_polar, 
                        kerrgeo_Vr, 
                        kerrgeo_Vtheta,
                        kerrgeo_Vphi_radial, 
                        kerrgeo_Vphi_polar)

from scipy.special import factorial as fac
from spherical import Wigner3j as w3j
from scipy.special import sph_harm, ellipk

import numpy as np

def muCoupling(s, l):
    """
    Eigenvalue for the spin-weighted spherical harmonic lowering operator
    Setting s -> -s gives the negative of the eigenvalue for the raising operator
    """
    if l + s < 0 or l - s + 1. < 0:
        return 0
    return np.sqrt((l - s + 1.)*(l + s))

def Asjlm(s, j, l, m):
    if s >= 0:
        return (-1.)**(m + s)*np.sqrt(4**s*fac(s)**2*(2*l + 1)*(2*j + 1)/fac(2*s))*w3j(s, l, j, 0, m, -m)*w3j(s, l, j, s, -s, 0)
    else:
        return (-1.)**(m)*np.sqrt(4**(-s)*fac(-s)**2*(2*l + 1)*(2*j + 1)/fac(-2*s))*w3j(-s, l, j, 0, m, -m)*w3j(-s, l, j, s, -s, 0)

def spin_operator_normalization(s, ns, l):
    s_sgn = np.sign(s)
    nmax1 = np.abs(s) + 1
    Jterm = 1.
    for ni in range(1, ns + 1):
        Jterm *= -s_sgn*muCoupling((nmax1-ni), l)
    return Jterm

def full_libration(xp):
    return np.concatenate((xp, np.flip(xp[1:-1])))

def construct_hertz_radial_components(hertz, samples = 2**6):
    samples_half = int(samples/2 + 1)
    hertzR0In = np.array([hertz.radialsolution('In', i) for i in range(samples_half)])
    hertzR0Up = np.array([hertz.radialsolution('Up', i) for i in range(samples_half)])
    hertzR1In = np.array([hertz.radialderivative('In', i) for i in range(samples_half)])
    hertzR1Up = np.array([hertz.radialderivative('Up', i) for i in range(samples_half)])
    hertzR2In = np.array([hertz.radialderivative2('In', i) for i in range(samples_half)])
    hertzR2Up = np.array([hertz.radialderivative2('Up', i) for i in range(samples_half)])
    hertzR = np.array(
        [[np.concatenate((hertzR0In, np.flip(hertzR0In)[1:-1])), np.concatenate((hertzR0Up, np.flip(hertzR0Up)[1:-1]))], 
        [np.concatenate((hertzR1In, np.flip(hertzR1In)[1:-1])), np.concatenate((hertzR1Up, np.flip(hertzR1Up)[1:-1]))], 
        [np.concatenate((hertzR2In, np.flip(hertzR2In)[1:-1])), np.concatenate((hertzR2Up, np.flip(hertzR2Up)[1:-1]))]])
    return hertzR

def construct_hertz_polar_spin_weighting(hertz, samples = 2**6):
    samples_half = int(samples/2 + 1)
    th = np.array([hertz.polarpoint(i) for i in range(samples_half)])
    thp = full_libration(th)
    sth = np.sin(thp)
    spinWeightingPolar = np.array(
        [pow(sth, -2),
        pow(sth, -1),
        pow(sth, 0)]
    )
    return spinWeightingPolar

def convergence_check(value, previous_value, ref_value, tol = 1e-8):
    if (np.abs(value) + np.abs(previous_value)) < 1e-10*np.abs(ref_value):
        return True
    elif np.abs(value) < tol*np.abs(ref_value) and np.abs(previous_value) < tol*np.abs(ref_value) and np.abs(value) < np.abs(previous_value):
        return True
    return False

def ut_ecc(q, En, Lz, r):
    Sigma = r**2
    theta = 0.5*np.pi
    Qc = 0.
    return (np.array([kerrgeo_Vt_radial(q, En, Lz, Qc, ri) for ri in r]) + kerrgeo_Vt_polar(q, En, Lz, Qc, theta))/Sigma

def ur_ecc(q, En, Lz, r):
    r0 = np.unique(r)
    Sigma = r**2
    Qc = 0.
    urUp = np.sqrt(np.abs(np.array([kerrgeo_Vr(q, En, Lz, Qc, ri) for ri in r0])))
    urUp[0] = 0.
    urUp[-1] = 0.
    urDown = -np.flip(urUp)[1:-1]
    ur = np.concatenate((urUp, urDown))/Sigma
    return ur

def uphi_ecc(q, En, Lz, r):
    Sigma = r**2
    theta = 0.5*np.pi
    Qc = 0.
    return (np.array([kerrgeo_Vphi_radial(q, En, Lz, Qc, ri) for ri in r]) + kerrgeo_Vphi_polar(q, En, Lz, Qc, theta))/Sigma

def ut_ecc_geo(geo):
    dM = geo.orbitalenergy
    dJ = geo.orbitalangularmomentum
    r = np.concatenate((geo.radialpoints, np.flip(geo.radialpoints)[1:-1]))
    q = geo.blackholespin
    return ut_ecc(q, dM, dJ, r)

def ur_ecc_geo(geo):
    dM = geo.orbitalenergy
    dJ = geo.orbitalangularmomentum
    r = np.concatenate((geo.radialpoints, np.flip(geo.radialpoints)[1:-1]))
    q = geo.blackholespin
    return ur_ecc(q, dM, dJ, r)

def uphi_ecc_geo(geo):
    dM = geo.orbitalenergy
    dJ = geo.orbitalangularmomentum
    r = np.concatenate((geo.radialpoints, np.flip(geo.radialpoints)[1:-1]))
    q = geo.blackholespin
    return uphi_ecc(q, dM, dJ, r)

def completion_ecc(geo):
    dM = geo.orbitalenergy
    dJ = geo.orbitalangularmomentum
    r = np.concatenate((geo.radialpoints, np.flip(geo.radialpoints)[1:-1]))
    q = geo.blackholespin
    htt = 2*dM/r
    hrr = 2.*r**2*((r + q**2)*dM - q*dJ)/(r**2 - 2.*r + q**2)**2
    htphi = -2*dJ/r
    hphiphi = 2.*q*((r + 2.)*dJ - (r + 1)*q*dM)/r
    ut = ut_ecc(q, dM, dJ, r)
    uphi = uphi_ecc(q, dM, dJ, r)
    ur = ur_ecc(q, dM, dJ, r)
    # return np.mean(r**2*(htt*ut*ut + 2*htphi*ut*uphi + hrr*ur*ur + hphiphi*uphi*uphi))/np.mean(r**2)
    return (htt*ut*ut + 2*htphi*ut*uphi + hrr*ur*ur + hphiphi*uphi*uphi)

def completion_ecc_v2(geo):
    dM = geo.orbitalenergy
    dJ = geo.orbitalangularmomentum
    r = np.concatenate((geo.radialpoints, np.flip(geo.radialpoints)[1:-1]))
    q = geo.blackholespin

    htt_dM = 2/r
    htt_dJ = 0
    htt = htt_dM*dM + htt_dJ*dJ

    hrr_dM = 2.*r**2*(r + q**2)/(r**2 - 2.*r + q**2)**2
    hrr_dJ = -2.*r**2*q/(r**2 - 2.*r + q**2)**2
    hrr = hrr_dM*dM + hrr_dJ*dJ

    htphi_dM = 0
    htphi_dJ = -2/r
    htphi = htphi_dM*dM + htphi_dJ*dJ

    hphiphi_dM = -2.*q**2*(r + 1)/r
    hphiphi_dJ = 2.*q*(r + 2.)/r
    hphiphi = hphiphi_dM*dM + hphiphi_dJ*dJ

    ut = ut_ecc(q, dM, dJ, r)
    uphi = uphi_ecc(q, dM, dJ, r)
    ur = ur_ecc(q, dM, dJ, r)
    return (htt*ut*ut + 2*htphi*ut*uphi + hrr*ur*ur + hphiphi*uphi*uphi)

def z0_average_geo(geo):
    dM = geo.orbitalenergy
    dJ = geo.orbitalangularmomentum
    r = np.concatenate((geo.radialpoints, np.flip(geo.radialpoints)[1:-1]))
    q = geo.blackholespin
    return np.mean(ut_ecc(q, dM, dJ, r)*r**2)/np.mean(r**2)

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

def huu_reg_inc(geo):
    En = geo.orbitalenergy
    Lz = geo.orbitalangularmomentum
    Qc = geo.carterconstant
    r = geo.radialpoints[0]
    th0 = np.concatenate((geo.polarpoints, np.flip(geo.polarpoints)[1:-1]))
    q = geo.blackholespin
    return huu_reg_gen_base(q, En, Lz, Qc, r, th0)

def huu_reg_ecc(geo):
    En = geo.orbitalenergy
    Lz = geo.orbitalangularmomentum
    Qc = geo.carterconstant
    r = geo.radialpoints
    r0 = full_libration(r)
    q = geo.blackholespin
    return huu_reg_gen_base(q, En, Lz, Qc, r0, 0.5*np.pi)

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

def orbit_average_inc(func, geo, axis = 0):
    th = np.concatenate((geo.polarpoints, np.flip(geo.polarpoints)[1:-1])).T
    r = geo.radialpoints[0]
    q = geo.blackholespin
    sig = r**2 + q**2*np.cos(th)**2
    if len(func.shape) <= 1:
        sigA = np.array(sig)
    else:
        sigA = np.array([sig, sig]).T
    return np.mean(func*sigA, axis = axis)/np.mean(sig)

def orbit_average_ecc(func, geo, axis = 0):
    r = np.concatenate((geo.radialpoints, np.flip(geo.radialpoints)[1:-1]))
    return np.mean(func*r**2, axis = axis)/np.mean(r**2)

def choose_spin_from_gauge(gauge):
    if gauge == "ORG" or gauge == "SAAB4" or gauge == "ASAAB4":
        s = 2
    else:
        s = -2
    return s

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

import os
import json

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

    # check size
    data_file = os.path.join(data_dir, f"huuYl-{jobname}-km-m{0}.npy")
    data_tmp = np.load(data_file)
    if data_tmp.size == 4*(lmax+1)*nsamples*nsamples:
        huuYl_arr = np.zeros((4, lmax+1, nsamples, nsamples))
    elif data_tmp.size == 4*(lmax+1)*nsamples:
        huuYl_arr = np.zeros((4, lmax+1, nsamples))
    else:
        huuYl_arr = np.zeros((4, lmax+1))

    for m in range(0, lmax + 1):
        try:
            data_file = os.path.join(data_dir, f"huuYl-{jobname}-km-m{m}.npy")
            data_lm = np.load(data_file).swapaxes(0,1).reshape(huuYl_arr.shape)
            huuYl_arr += data_lm
        except FileNotFoundError:
            pass
            # print(f"huuYl-{jobname}-km-m{m}.npy not found")
        try:
            data_file = os.path.join(data_dir, f"huuYl-{jobname}-kp-m{m}.npy")
            data_lm = np.load(data_file).swapaxes(0,1).reshape(huuYl_arr.shape)
            huuYl_arr += data_lm
        except FileNotFoundError:
            pass
            # print(f"huuYl-{jobname}-kp-m{m}.npy not found")
        try:
            data_file = os.path.join(data_dir, f"metadata-{jobname}-km-m{m}.txt")
            data_meta = np.loadtxt(data_file)
            meta_data.append(data_meta)
        except FileNotFoundError:
            pass
            # print(f"metadata-{jobname}-km-m{m}.txt not found")
        try:
            data_file = os.path.join(data_dir, f"metadata-{jobname}-kp-m{m}.txt")
            data_meta = np.loadtxt(data_file)
            meta_data.append(data_meta)
        except FileNotFoundError:
            pass
            # print(f"metadata-{jobname}-km-m{m}.txt not found")
    
    return huuYl_arr[:2], params, meta_data

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
    mean_flag = False
    if huuYl.shape[0] == 2:
        lmax = huuYl.shape[1] - 1
        mean_flag = True
    else:
        lmax = huuYl.shape[0] - 1
    datareg_avg = orbit_average_gen(huuYl - huuReg, geo, axis = axis)
    if mean_flag:
        datareg_avg = np.mean(datareg_avg, axis = 0)
    huu_fits = []
    for lmax_test in range(min_lmax, lmax):
        b = np.array([lfit(datareg_avg, lmin_var, lmax_test, nmin = nmin, nmax = nmax) for lmin_var in range(lmin_test, lmax_test)])
        huu_fits.append([np.mean(b[:, 0], axis = 0), 2*np.std(b[:, 0], axis = 0), lmax_test, *np.mean(b[:, 1:], axis = 0)])
    huu_fits = np.array(huu_fits)
    huu_comp = orbit_average_gen(completion_gen(geo), geo)
    best_fit = huu_fits[np.argmin(huu_fits[:, 1], axis = 0)]
    huu_recon = best_fit[0]
    huu_err = best_fit[1]
    lmax_test = int(np.abs(best_fit[2]))
    Hn_fits = best_fit[3:]
    huu_lmax = datareg_avg[lmax_test]
    for n in range(nmin, nmax):
        huu_lmax *= Hn_fits[n - nmin]*lweight(lmax_test, n)
    huu_val = huu_recon + huu_comp
    z0 = z0_gen_geo(geo)
    z1 = -0.5*z0*huu_val
    z1_err = 0.5*z0*huu_err

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

    return {"z0": z0, 
            "fits": huu_fits,
            "completion": huu_comp,
            "est1": {"z1": z1, "err": z1_err, "lmax": lmax_test, "z1_lmax": -0.5*z0*huu_lmax},
            "est2": {"z1": z1_2, "err": z1_err_2, "lmax": lmax_test_2, "z1_lmax": -0.5*z0*huu_lmax_2},
            }

import pandas as pd
if __name__ == "__main__":
    data_dir = "data"
    df = pd.read_csv(os.path.join(data_dir, "huu_metadata.csv"))
    
    data_list = []
    z1_str_list = []
    paper_iter_list = []
    max_iter = len(df)

    for i in range(max_iter):
        gauge = df.loc[i, 'gauge']
        name = df.loc[i, 'name']

        print(f"Processing {gauge} data for {name}...")
        print("==========================")

        params_file = df.loc[i, 'filename_params']

        with open(os.path.join(data_dir, params_file)) as json_file:
            params = json.load(json_file)
        huuYl_arr = np.load(os.path.join(data_dir, df.loc[i, 'filename_huu']))

        a = params['a']
        p = params['p']
        e = params['e']
        x = params['x']
        print(a,p,e,x)

        geo = KerrGeodesic(a, p, e, x, nsamples = (params["nsamples"] // 1))
        geo2 = KerrGeodesic(a, p, e, x, nsamples = 2*(params["nsamples"] // 1))
        # print(1-z0_gen_geo(geo)/z0_gen_geo(geo2))
        
        huuYl_test = np.mean(huuYl_arr, axis = 0)
        # huuYl_test = huuYl_arr[1]

        if e == 0.:
            huuYl_test = huuYl_test[:, np.newaxis, :]
            huureg = huu_reg_inc(geo)[np.newaxis, :]
        else:
            huureg = huu_reg_gen(geo)

        out = huu_fit(huuYl_test, huureg, geo, lmin_test = 5, min_lmax = 12, axis = 2, nmax = 3)
        z1 = out["est1"]["z1"]
        z1_err = out["est1"]["err"]
        z1_lmax = out["est1"]["z1_lmax"]
        z0 = out["z0"]
        lmax_cut = out["est1"]["lmax"]
        data = [gauge, name, a, p, e, x, z0, z1, z1_err, lmax_cut, z1_lmax]
        data_list.append(data)
        print(data)

        # Save the results
        output_dir = f"results/{gauge}/{name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f"z1-{gauge}-{name}.npy")
        np.save(output_file, data)
        print(f"Results saved to {output_file}")

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

        z1_str_list.append(f"{z1_trunc}({error_digit})")
        paper_iter_list.append(i)

    df_data = pd.DataFrame(data_list, columns=['gauge', 'name', 'a', 'p', 'e', 'x', 'z0', 'z1', 'z1_err', 'lmax_cut', 'z1_lmax'])
    df_data.sort_values(by=['a', 'e', 'name', 'gauge'], inplace=True, ignore_index=True)
    df_data.to_csv(
        f"results/z1-all.csv"
    )
    for i, z1_str in enumerate(z1_str_list):
        print(paper_iter_list[i], data_list[i][0], data_list[i][1], data_list[i][2], data_list[i][3], z1_str)