# -*— coding: utf-8 -*_
# @Time    : 2020/12/29 5:01 下午
# @Author  : Algo
# @Site    : 
# @File    : das_srp.py
# @Software: PyCharm
import numpy as np
import tqdm
from matplotlib import pyplot as plt
theta = range(0, 360)
def array_location(inputsignal, mic_number, mic_distance, c, fs):
    N = inputsignal.shape[0]
    X_fft = np.fft.rfft(inputsignal, axis=0)
    omega = 2 * np.pi * np.tile(np.arange(N/2+1), (mic_number, 1)).T * fs/N
    array_p = []
    for i in tqdm.tqdm(theta[:180]):
        tau = np.arange(mic_number) * mic_distance * np.cos(np.deg2rad(i))/c
        array_y = np.sum(X_fft * np.exp(-1j * omega * tau), axis=1)
        array_p.append(np.linalg.norm(array_y))
    array_k = np.argmax(array_p)
    print(array_k, -array_k % 360)
    plt.plot(theta, array_p+array_p[::-1])
    plt.show()
def mirror_location(inputsignal, mic_number, mic_distance, c, fs):
    mirror_n = inputsignal.shape[0]
    mirrorx_fft = np.fft.rfft(inputsignal, axis=0)
    omega = 2 * np.pi * np.tile(np.arange(mirror_n / 2 + 1), (mic_number, 1)).T * fs / mirror_n
    mirror_p = []
    for i in tqdm.tqdm(theta):
        tau = mic_distance * np.cos(np.deg2rad(i - np.arange(mic_number) / mic_number *360))/c
        mirror_y = np.sum(mirrorx_fft * np.exp(-1j * omega * tau), axis=1)
        mirror_p.append(np.linalg.norm(mirror_y))
    mirror_k = np.argmax(mirror_p)
    print(mirror_k)
    plt.plot(theta, mirror_p)
    plt.show()



