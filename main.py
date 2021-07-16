import numpy as np
import librosa
import os
import das_srp
import srp_phat
voice_speed = 340
array_mic_num = 6
array_mic_dis = 0.05
mirror_mic_num = 6
mirror_mic_dis = 0.05
fs = 16000


def read_waves(type):
    root = os.getcwd()
    file_root = os.path.join(root, "wav")
    files = os.listdir(file_root)
    wavelist = []
    x = []
    for file in files:
        name, category = os.path.splitext(os.path.join(file_root, file))
        if type in name and category == '.wav':  
            wavelist.append(file)
    wavelist.sort(key=lambda x: int(x[0]))
#    print(wavelist)
    for wav in wavelist:
        x.append(librosa.load(os.path.join(file_root, wav), sr=None)[0])
    return x


if __name__ == '__main__':
    x_array = np.array(read_waves("array"))
    das_srp.array_location(x_array.T, array_mic_num, array_mic_dis, voice_speed, fs)
    srp_phat.array_location(x_array.T, array_mic_num, array_mic_dis, voice_speed, fs)
    x_mirror = np.array(read_waves("mirror"))
    das_srp.mirror_location(x_mirror.T, mirror_mic_num, mirror_mic_dis, voice_speed, fs)
    srp_phat.mirror_location(x_mirror.T, mirror_mic_num, mirror_mic_dis, voice_speed, fs)
