import matplotlib.pylab as plt
import librosa
import numpy as np

from waveletFunctions import wave_signif, wavelet

#TODO check if lag and multiplier can be better
def analyseAudio(PATH, sr):
    # READ THE DATA
    sst, sample_rate = librosa.load(PATH, sr=sr)
    variance = np.std(sst, ddof=1) ** 2

    if 0:
        variance = 1.0
        sst = sst / np.std(sst, ddof=1)
    n = len(sst)
    dt = 1 / sample_rate
    pad = 1  # pad the time series with zeroes (recommended)
    dj = 0.05  # this will do 4 sub-octaves per octave
    s0 = 10 * dt  # this says start at a scale of 6 months
    j1 = 5 / dj  # this says do 7 powers-of-two with dj sub-octaves each
    lag1 = 0.9  # lag-1 autocorrelation for red noise background
    mother = 'MORLET'

    white_noice_multiplier = 1.2

    # Wavelet transform:
    wave, rpms, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum

    # Significance levels:
    signif = wave_signif(variance * white_noice_multiplier, dt=dt, sigtest=0, scale=scale,
        lag1=lag1, mother=mother)
    
    # expand signif --> (J+1)x(N) array
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
    sig95 = power / sig95  # where ratio > 1, power is significant

    return sig95, rpms

def calculateRPMS(path, sr):
    #TODO: Make method great. Allow user to input some parameters, the user maybe dont want to get information of every second
    sig95, rpms = analyseAudio(path, sr)

    length = len(sig95[0])
    seconds = int(np.ceil(length/sr))

    intensitys = []
    rpmsMessured = []

    top = sr
    bottom = 0
    
    for s in range(0, seconds):
        intensity = 0
        maxIndex = 0
        for i in range(0, len(sig95)):
            current = np.max(sig95[i][bottom:top])
            if current > intensity:
                intensity = current
                maxIndex = i

        intensitys.append(intensity * 1000)
        rpmsMessured.append(rpms[maxIndex])

        top += sr
        bottom += sr

    return rpmsMessured, intensitys



def main():
    rpmsMessured, intensitys = calculateRPMS("audio/16-11-08.wav", 8000)
    plt.plot(rpmsMessured)
    plt.xlabel('Time (seconds)')
    plt.ylabel('RPM')
    plt.plot(intensitys, '--')
    plt.title("6000 - Air")
    plt.show()

if __name__ == '__main__':
    main()
    