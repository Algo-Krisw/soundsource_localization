# -*— coding: utf-8 -*_
# @Time    : 2020/12/29 11:16 下午
# @Author  : Algo
# @Site    : 
# @File    : srp_phat.py
# @Software: PyCharm
import numpy as np
import tqdm
from matplotlib import pyplot as plt
theta = range(0, 360)
def array_location(inputsignal, mic_number, mic_distance, c, fs):
    N = inputsignal.shape[0]
    X_fft = np.fft.rfft(inputsignal, axis=0)
    omega = 2 * np.pi * np.arange(N/2 + 1) * fs/N
    array_p = [0] * 180
    for i in tqdm.tqdm(theta[:180]):
        tau = np.arange(mic_number) * mic_distance * np.cos(np.deg2rad(i))/c
        tau = np.tile(tau, (mic_number, 1)) - tau[:, np.newaxis]
        for m in range(mic_number):
            for n in range(mic_number):
                XX = X_fft[:, m] * np.conj(X_fft[:, n])
                XX = XX / np.abs(XX) * np.exp(-1j * omega * tau[n, m])
                array_p[i] += np.sum(XX).real
    array_k = np.argmax(array_p)
    print(array_k, -array_k % 360)
    plt.plot(theta, array_p+array_p[::-1])
    plt.show()
def mirror_location(inputsignal, mic_number, mic_distance, c, fs):
    mirror_n = inputsignal.shape[0]
    mirrorx_fft = np.fft.rfft(inputsignal, axis=0)
    omega = 2 * np.pi * np.arange(mirror_n / 2 + 1) * fs / mirror_n
    mirror_p = [0] * 360
    for i in tqdm.tqdm(theta):
        tau = mic_distance * np.cos(np.deg2rad(i - np.arange(mic_number) / mic_number * 360))/c
        tau = np.tile(tau, (mic_number, 1)) - tau[:, np.newaxis]
        for m in range(mic_number):
            for n in range(mic_number):
                XX = mirrorx_fft[:, m] * np.conj(mirrorx_fft[:, n])
                XX = XX / np.abs(XX) * np.exp(-1j * omega * tau[n, m])
                mirror_p[i] += np.sum(XX).real
    mirror_k = np.argmax(mirror_p)
    print(mirror_k)
    plt.plot(theta, mirror_p)
    plt.show()