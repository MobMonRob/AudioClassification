import numpy as np

from cwtUtilsRPM import wave_signif, wavelet

def analyseAudio(data, sample_rate):
    variance = np.std(data, ddof=1) ** 2

    if 0:
        variance = 1.0
        data = data / np.std(data, ddof=1)

    n = len(data)
    dt = 1 / sample_rate
    pad = 1  # pad the time series with zeroes (recommended)

    # These parameters control the filter which is used für the wavelet funcion and controls the area of frequencies that are analysed
    # For this extraction a inversed red-noise-filter is used, this is defined in cwtUtilsGrindingStatus line 355
    dj = 0.05
    s0 = 10 * dt
    j1 = 5 / dj
    lag1 = 0.9
    mother = 'MORLET'

    white_noice_multiplier = 1.2

    # Wavelet transform:
    wave, rpms, scale, coi = wavelet(data, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum

    # Significance levels:
    signif = wave_signif(variance * white_noice_multiplier, dt=dt, sigtest=0, scale=scale,
        lag1=lag1, mother=mother)
    
    # expand signif --> (J+1)x(N) array
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
    sig95 = power / sig95

    return sig95, rpms



def calculateRPMS(data, sr):
    sig95, rpms = analyseAudio(data, sr)

    length = len(sig95[0])
    seconds = int(np.floor(length/sr))

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

        intensitys.append(intensity)
        rpmsMessured.append(rpms[maxIndex])

        top += sr
        bottom += sr

    return rpmsMessured, intensitys