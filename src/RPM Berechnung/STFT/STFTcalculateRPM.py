import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
import librosa


import numpy as np

def analyseAudio(path, sr):
    audio, sr = librosa.load(path, sr = sr)

    N = len(audio)
    T_x = 1 / sr
    t_x = np.arange(N) * T_x
    dr = 1 / T_x

    g_std = sr/100
    w = gaussian(4*sr, std=g_std, sym=True)  # symmetric Gaussian window

    SFT = signal.ShortTimeFFT(w, int(sr/10), dr, mfft=128000, scale_to='magnitude', fft_mode='onesided')

    Sx = SFT.stft(audio)

    return Sx, SFT

def calculateRPMS(path, sr):
    Sx, SFT = analyseAudio(path, sr)

    slices = len(Sx[0])
    seconds = int(SFT.delta_t * slices)

    rpmsMessured = []
    intensitysMessured = []

    top = 10
    bot = 0

    for i in range(0 ,seconds):
        #get highest frequnen in first second
        freq = 0
        intensity = 0
        for n in range(0, len(Sx)):
            currentIntensity = np.max(Sx[n][bot:top])
            if currentIntensity > intensity:
                freq = n
                intensity = currentIntensity

        rpmsMessured.append(freq * SFT.delta_f * 60)
        intensitysMessured.append(intensity * 100000)

        bot += 10
        top += 10
    

    return rpmsMessured, intensitysMessured



def main():
    rpms, intensitys = calculateRPMS("audio/16-11-04.wav", 8000)
    
    fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit

    ax1.set_title("6000 - Grinding")
    ax1.set(xlabel=f"Time $t$ in seconds",
        ylabel=f"RPM",)
    plt.plot(rpms)
    plt.plot(intensitys,'--')
    plt.legend(['RPM', 'Intensity'])
    plt.show()

if __name__ == '__main__':
    main()
