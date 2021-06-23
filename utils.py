# -*- coding: utf-8 -*-
# @Time    : 4/14/21 7:43 PM
# @Author  : Yan
# @Site    : 
# @File    : utils.py
# @Software: PyCharm

import os
import argparse
from typing import List
import h5py
import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft, ifft, rfft, irfft, rfftfreq
from matplotlib import mlab


def read_h5(path):
    f = h5py.File(path, 'r')
    for key in f.keys():
        print(f[key].name)
        print(f[key].shape)
        # print(f[key].value)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-u", "--update_data", type=bool, default=False)
    parser.add_argument("-n", "--num_runs", type=int, default=10)
    parser.add_argument("-k", "--num_kernels", type=int, default=10_000)
    parser.add_argument("-s", "--if_split", type=bool, default=False)
    return parser.parse_args()


def file2list(path):
    with open(path, "r") as f:
        lines = [int(line.strip()) for line in f]
    return lines


def list2file(path, list_a):
    with open(path, "w") as f:
        for item in list_a:
            f.write("%s\n" % str(item))


def normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def check_if_file_exits(file_name):
    return os.path.exists(file_name)


def delete_history(path):
    if isinstance(path, list):
        for path_string in path:
            if check_if_file_exits(path_string):
                os.remove(path_string)
    if isinstance(path, str):
        if check_if_file_exits(path):
            os.remove(path)


def center(X):
    newX = X - np.mean(X, axis=0)
    return newX


def zca_whitening(inputs):
    if isinstance(inputs, List):
        inputs = np.array(inputs)
    sigma = np.dot(inputs, inputs.T) / (inputs.shape[0] - 1)
    U, S, V = np.linalg.svd(sigma)
    epsilon = 0.1
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(np.diag(S) + epsilon))), U.T)  # 计算zca白化矩阵
    return np.dot(ZCAMatrix, inputs)


def interp_psd(freqs, ps, lF, uF):
    idx = np.argsort(freqs)  # sorting frequencies

    sum = 0
    c = 0
    for i in idx:
        if (freqs[i] >= lF) and (freqs[i] <= uF):
            sum += ps[i]
            c += 1
    return sum / c


def whiten(strain, dt=1, phase_shift=0, time_shift=0):
    """Whitens strain data given the psd and sample rate, also applying a phase
    shift and time shift.

    Args:
        strain (ndarray): strain data
        interp_psd (interpolating function): function to take in freqs and output
            the average power at that freq
        dt (float): sample time interval of data
        phase_shift (float, optional): phase shift to apply to whitened data
        time_shift (float, optional): time shift to apply to whitened data (s)

    Returns:
        ndarray: array of whitened strain data
    """
    Nt = len(strain)
    ps = np.abs(fft(strain)) ** 2
    # take the fourier transform of the data
    freqs = rfftfreq(Nt, dt)

    # whitening: transform to freq domain, divide by square root of psd, then
    # transform back, taking care to get normalization right.
    hf = rfft(strain)

    # apply time and phase shift
    hf = hf * np.exp(-1.j * 2 * np.pi * time_shift * freqs - 1.j * phase_shift)
    norm = 1. / np.sqrt(1. / (dt * 2))
    white_hf = hf / np.sqrt(interp_psd(freqs, ps, 0, 200)) * norm
    # white_ht = irfft(white_hf)
    white_ht = np.real(white_hf)
    return white_ht


def bandpass(strain, fband, fs):
    """Bandpasses strain data using a butterworth filter.

    Args:
        strain (ndarray): strain data to bandpass
        fband (ndarray): low and high-pass filter values to use
        fs (float): sample rate of data

    Returns:
        ndarray: array of bandpassed strain data
    """
    bb, ab = butter(4, [fband[0] * 2. / fs, fband[1] * 2. / fs], btype='band')
    normalization = np.sqrt((fband[1] - fband[0]) / (fs / 2))
    strain_bp = filtfilt(bb, ab, strain) / normalization
    return strain_bp.tolist()


def get_full_psds(input_time_series):
    fs = int(1.0 / 0.5)

    # number of sample for the fast fourier transform:
    NFFT = 4 * fs  # Use 4 seconds of data for each fourier transform
    NOVL = 1 * NFFT / 2  # The number of points of overlap between segments used in Welch averaging
    psd_window = scipy.signal.tukey(NFFT, alpha=1. / 4)

    Pxx_H1, freqs = mlab.psd(input_time_series, Fs=fs, NFFT=NFFT,
                             window=psd_window, noverlap=NOVL)
    psd_H1 = interp1d(freqs, Pxx_H1)
    return psd_H1

# def get_full_psds(eventnames, large_data_filenames,
#                   make_plots=False, plot_others=False):
#     """Obtains full 1024 second psds for all the events specified. Uses the Welch
#     average technique, along with other less accurate techniques if
#     specified. Can also plot the psd obtained.
#
#     Args:
#         eventnames (list): list of events to get psds for
#         large_datafilenames (dict): dictionary whose keys are the eventnames
#             and whose values are the filenames of the large amounts of strain
#             data used, without the added 'H-<det>_'
#         make_plots (bool, optional): if set to True, plot psd data
#         plot_others (bool, optional): if set to True, also obtain psd data
#             without averaging as well as with no window
#
#     Returns:
#         dict: A dictionary containing psds for each detector for each event
#             specified in eventnames.
#     """
#
#     large_data_psds = {}
#     for eventname in eventnames:
#         large_data_psds[eventname] = {'H1': [], 'L1': []}
#
#         # get filename
#         fn_H1 = 'H-H1_' + large_data_filenames[eventname]
#         fn_L1 = 'L-L1_' + large_data_filenames[eventname]
#
#         # get strain data
#         strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
#         strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')
#
#         # both H1 and L1 will have the same time vector, so:
#         time = time_H1
#
#         indxt_around = np.where((time >= time_center - 512) & (
#                 time < time_center + 512))
#
#         # number of sample for the fast fourier transform:
#         NFFT = 4 * fs  # Use 4 seconds of data for each fourier transform
#         NOVL = 1 * NFFT / 2  # The number of points of overlap between segments used in Welch averaging
#         psd_window = scipy.signal.tukey(NFFT, alpha=1. / 4)
#
#         Pxx_H1, freqs = mlab.psd(strain_H1[indxt_around], Fs=fs, NFFT=NFFT,
#                                  window=psd_window, noverlap=NOVL)
#         Pxx_L1, freqs = mlab.psd(strain_L1[indxt_around], Fs=fs, NFFT=NFFT,
#                                  window=psd_window, noverlap=NOVL)
#
#         if (plot_others):
#             # smaller window if we're not doing Welch's method
#             short_indxt_away = np.where((time >= time_center - 2) & (
#                     time < time_center + 2))
#             # psd using a tukey window but no welch averaging
#             tukey_Pxx_H1, tukey_freqs = mlab.psd(
#                 strain_H1[short_indxt_away], Fs=fs, NFFT=NFFT, window=psd_window)
#             # psd with no window, no averaging
#             nowin_Pxx_H1, nowin_freqs = mlab.psd(
#                 strain_H1[short_indxt_away], Fs=fs, NFFT=NFFT,
#                 window=mlab.window_none)
#
#         # We will use interpolations of the PSDs computed above for whitening:
#         psd_H1 = interp1d(freqs, Pxx_H1)
#         psd_L1 = interp1d(freqs, Pxx_L1)
#
#         large_data_psds[eventname]['H1'] = psd_H1
#         large_data_psds[eventname]['L1'] = psd_L1
#
#         if make_plots:
#             plt.figure(figsize=(8, 5))
#             # scale x and y axes
#             plt.xscale('log', basex=2)
#             plt.yscale('log', basey=10)
#
#             # plot nowindow, tukey, welch together
#             plt.plot(nowin_freqs, nowin_Pxx_H1, 'purple', label='No Window',
#                      alpha=.8, linewidth=.5)
#             plt.plot(tukey_freqs, tukey_Pxx_H1, 'green', label='Tukey Window',
#                      alpha=.8, linewidth=.5)
#             plt.plot(freqs, Pxx_H1, 'black', label='Welch Average', alpha=.8,
#                      linewidth=.5)
#
#             # plot 1/f^2
#             # give it the right starting scale to fit with the rest of the plots
#             # don't include zero frequency
#             inverse_square = np.array(list(map(lambda f: 1 / (f ** 2),
#                                                nowin_freqs[1:])))
#             # inverse starts at 1 to take out 1/0
#             scale_index = 500  # chosen by eye to fit the plot
#             scale = nowin_Pxx_H1[scale_index] / inverse_square[scale_index]
#             plt.plot(nowin_freqs[1:], inverse_square * scale, 'red',
#                      label=r'$1 / f^2$', alpha=.8, linewidth=1)
#
#             plt.axis([20, 512, 1e-48, 1e-41])
#             plt.ylabel('Sn(t)')
#             plt.xlabel('Freq (Hz)')
#             plt.legend(loc='upper center')
#             plt.title('LIGO PSD data near ' + eventname + ' at H1')
#             plt.show()
#
#     return large_data_psds

