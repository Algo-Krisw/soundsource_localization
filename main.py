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
_, fs = librosa.load("1_array.wav", sr=None)
def read_waves(type):
    filepath = "."
    filenames = os.listdir(filepath)
    wavelist = []
    x = []
    for filename in filenames:
        name, category = os.path.splitext(filepath + filename)
        if type in name and category == '.wav':  
            wavelist.append(filename)
    wavelist.sort(key=lambda x: int(x[0]))
    print(wavelist)
    for wav in wavelist:
        x.append(librosa.load(wav, sr=None)[0])
    return x
if __name__ == '__main__':
    x_array = np.array(read_waves("array"))
    das_srp.array_location(x_array.T, array_mic_num, array_mic_dis, voice_speed, fs)
    srp_phat.array_location(x_array.T, array_mic_num, array_mic_dis, voice_speed, fs)
    x_mirror = np.array(read_waves("mirror"))
    das_srp.mirror_location(x_mirror.T, mirror_mic_num, mirror_mic_dis, voice_speed, fs)
    srp_phat.mirror_location(x_mirror.T, mirror_mic_num, mirror_mic_dis, voice_speed, fs)
