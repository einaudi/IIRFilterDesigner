# -*- coding: utf-8 -*-

import numpy as np

from scipy.optimize import curve_fit
from scipy.special import binom
from scipy.signal import iirfilter, freqz


# Conversions
def att_to_dB(att):

    return 10*np.log10(att)

def dB_to_att(A):

    ret = -np.absolute(A)/10.
    ret = np.power(10., ret)

    return ret

def warp_frequency(omega_pre, f_samp):

    return 2*np.arctan(0.5*omega_pre/f_samp)

def prewarp_frequency(omega, f_samp):

    return 2*f_samp * np.tan(omega/2)


# Initial calculations
def calc_order(Omega_p, Omega_s, att_p, att_s, btype='lowpass', mode='Butterworth'):

    # frequency mapping
    Omega_p_map = map_frequency(Omega_p, 1, btype)
    Omega_s_map = map_frequency(Omega_s, 1, btype)

    if mode == 'Butterworth':
        nom = np.power(att_s, -2) - 1
        nom /= np.power(att_p, -2) - 1
        nom = np.log10(nom)

        denom = Omega_s_map/Omega_p_map
        denom = 2*np.log10(denom)

        ret = nom/denom
        ret = np.ceil(ret)

        return int(ret)
    else:
        print('Order calculation. Unknown filter type.')
        return None
    
def calc_cutoff_freq(Omega, att, order, btype='lowpass', mode='Butterworth'):

    if mode == 'Butterworth':
        if btype == 'lowpass':
            ret = np.power(att, -2) - 1
            ret = np.power(ret, 1/2/order)
            ret = Omega/ret
        elif btype == 'highpass':
            ret = np.power(att, -2) - 1
            ret = np.power(ret, 1/2/order)
            ret = Omega*ret

        return ret
    else:
        print('Cutoff frequency calculation. Unknown filter type.')
        return None


# Frequency mapping to different band types
def map_frequency(Omega, Omega_c, btype='lowpass'):

    if btype == 'lowpass':
        return Omega
    elif btype == 'highpass':
        return Omega_c/Omega
    else:
        return 0


# Normalised filter calculations
def calc_normalised_lowpass_roots(order):

    ret = []
    N = int(order)
    for k in range(N):
        coef = (2.*k+N+1)/N
        ret.append(np.exp(0.5j*np.pi*coef))

    return ret

def get_continuous_transfer_function(Omega_c, order, sk, btype='lowpass'):

    N = int(order)
    def H(s):
        if btype == 'lowpass':
            s_coef = s/Omega_c
        elif btype == 'highpass':
            s_coef = Omega_c/s
        ret = 1
        for k in range(N):
            ret /= (s_coef - sk[k])

        return ret
    
    return H

def get_digital_transfer_function(Omega_c, order, sk, f_samp, btype='lowpass'):

    N = int(order)
    # def H(z):
    #     z_coef = (1-1/z)/(1+1/z) * 2*f_samp
    #     ret = np.power(Omega_c, N)
    #     for k in range(N):
    #         ret /= (z_coef - sk[k]*Omega_c)

    #     return ret

    continuous_filter = get_continuous_transfer_function(Omega_c, N, sk, btype)

    def H(z):
        z_coef = (1-1/z)/(1+1/z) * 2*f_samp
        return continuous_filter(z_coef)
    
    return H


# Digital filter parameters
def fit_digital_filter(filter_func, omegas, order, bounds=(-2, 2)):

    # Direct fitting
    N = int(order)

    def fit_func(omega, *args):

        z = np.exp(1j*omega)
        nom = 0
        for k in range(N+1):
            nom += args[k]*np.power(z, -k)
        denom = 1
        for k in range(N):
            denom += args[N+1+k]*np.power(z, -k-1)

        ret = np.absolute(nom/denom)

        return ret
    
    zs = np.exp(1j*omegas)
    ys = np.absolute(filter_func(zs))
    p0 = np.ones(2*N+1)*0.5
    # p0[N] = 1
    popt, _ = curve_fit(
        fit_func,
        omegas,
        ys,
        p0=p0,
        bounds=bounds
    )
    filter_fitted = fit_func(omegas, *popt)

    ff_coefs = np.array(popt[:N+1])
    fb_coefs = -np.array(popt[N+1:])

    return fb_coefs, ff_coefs, filter_fitted

    # Denominator fitting, numerator from Newton's binomial
    # N = int(order)

    # # numerator - binomial expansion of (1 - z^-1)^N
    # ff_coefs = np.zeros(N+1)
    # for k in range(N+1):
    #     ff_coefs[k] = binom(N, k) * np.power(-1, k)

    # # denominator - fitting
    # def fit_func(omega, *args):
    #     ret = 0
    #     zs = np.exp(1j*omega)
    #     for k in range(N+1):
    #         ret += args[k] * np.power(zs, -k)

    #     return np.absolute(ret)
    
    # zs = np.exp(1j*omegas)
    # ys = np.absolute(1 - np.power(zs, -1))
    # ys = np.power(ys, N) / np.absolute(filter_func(zs))
    # p0 = np.ones(N+1) * 0.5

    # fb_coefs, _ = curve_fit(
    #     fit_func,
    #     omegas,
    #     ys,
    #     p0=p0,
    #     bounds=bounds
    # )

    # # gain normalisation
    # ff_coefs /= fb_coefs[0]
    # fb_coefs /= fb_coefs[0]

    # # fitted filter
    # nom = 0
    # for k in range(N+1):
    #     nom += ff_coefs[k] * np.power(zs, -k)
    # denom = fit_func(omegas, *fb_coefs)

    # filter_fitted = np.absolute(nom/denom)

    # return fb_coefs, ff_coefs, filter_fitted

def get_digital_filter_coefs(order, omega_c, omega_samp, btype='lowpass'):

    N = int(order)
    omega_nyquist = omega_samp/2
    Wn = omega_c / omega_nyquist # normalise frequency to nyquist frequency

    b, a = iirfilter(N, Wn, btype=btype)

    w, h = freqz(
        b,
        a
    )

    ff_coefs = b / a[0]
    fb_coefs = a / a[0]
    fb_coefs = -fb_coefs[1:]

    return fb_coefs, ff_coefs, [w, h]

def get_digital_filter_zpk(order, omega_c, omega_samp, btype='lowpass'):

    N = int(order)
    omega_nyquist = omega_samp/2
    Wn = omega_c / omega_nyquist # normalise frequency to nyquist frequency

    z, p, k = iirfilter(N, Wn, btype=btype, output='zpk')

    return z, p, k


# Ideal filters
def get_ideal_Butterworth_function(Omega_c, order):

    N = int(order)
    def H(omega):
        ret = np.power(omega/Omega_c, 2*N)
        ret += 1
        ret = 1/np.sqrt(ret)

        return ret
    
    return H


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from src.IIR import IIRFilter
    from alive_progress import alive_bar

    f_p = 10 # Hz
    f_s = 100 # Hz

    A_p = -2 # dBm
    A_s = -20 # dBm

    f_samp = 1e3 # Hz

    M = 100
    
    bounds = (-np.inf, np.inf)

    # Calculations
    # frequencies
    Omega_p = 2*np.pi*f_p
    Omega_s = 2*np.pi*f_s

    fs = np.logspace(np.log10(1), np.log10(f_samp/2), M)
    Omegas = 2*np.pi*fs 
    omegas_warped = warp_frequency(Omegas, f_samp)

    # attenuation
    att_p = dB_to_att(A_p)
    att_s = dB_to_att(A_s)

    # filter parameters
    N = calc_order(Omega_p, Omega_s, att_p, att_s)
    print('Filter order: {}'.format(N))
    Omega_c = calc_cutoff_freq(Omega_s, att_s, N)
    sk = calc_normalised_lowpass_roots(N)

    # continuous time filter
    H = get_continuous_transfer_function(Omega_c, N, sk)

    # Z transformed filter
    H_z = get_digital_transfer_function(Omega_c, N, sk, f_samp)

    # Fitting filter
    # fb_coefs, ff_coefs, ys_fitted = fit_digital_filter(H_z, omegas_warped, N, bounds=bounds)
    # print('Feedback coefs: ', fb_coefs)
    # print('Feedforward coefs: ', ff_coefs)

    # Get filter coefs with scipy
    fb_coefs, ff_coefs, filter_computed = get_digital_filter_coefs(N, Omega_c, 2*np.pi*f_samp)
    print('Feedback coefs: ', fb_coefs)
    print('Feedforward coefs: ', ff_coefs)
    z, p, k = get_digital_filter_zpk(N, Omega_c, 2*np.pi*f_samp)

    filter_computed[0] = prewarp_frequency(filter_computed[0], f_samp) / 2/np.pi
    filter_computed[1] = att_to_dB(np.absolute(filter_computed[1]))

    # test fitted filter
    iir_A = []
    
    print('Calculating iir filter answer...')
    with alive_bar(fs.size) as bar:
        for f in fs:
            T = 500/f
            dt = 1/f_samp
            ts = np.linspace(0, T, int(T/dt))
            filt = IIRFilter(ff_coefs, fb_coefs)
            signal_in = 0.5*np.sin(2*np.pi*f*ts)
            signal_out = []
            for s in signal_in:
                signal_out.append(filt.update(s))
            iir_A.append(np.amax(signal_out[-100:]) - np.amin(signal_out[-100:]))

            bar()
    iir_A = att_to_dB(iir_A)
    print('Done!')

    # Plotting
    ys = att_to_dB(np.absolute(H(1j*Omegas)))
    ys_z = att_to_dB(np.absolute(H_z(np.exp(1j*omegas_warped))))

    fig = plt.figure(111)
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    # ax.set_yscale('log')

    ax.plot(fs, ys, c='C0', label='continuous')
    ax.plot(fs, ys_z, c='C1', label='z-transformed')
    # ax.plot(fs, ys_fitted, c='C2', label='fitted')
    ax.plot(filter_computed[0], filter_computed[1], c='C2', label='computed')
    ax.plot(fs, iir_A, c='C3', label='implemented')

    ax.legend(loc=0)

    ax.plot(
        [f_p, f_s],
        [A_p, A_s],
        'o',
        linewidth=0,
        markersize=5,
        zorder=10,
        color='r'
    )

    fig.savefig('./test.png')