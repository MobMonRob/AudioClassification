import matplotlib.pylab as plt
import librosa
import numpy as np

from cwtUtilGrindingStatus import wave_signif, wavelet

# Uses a Wavelet transform and a significance test to extract the relevant frequencies for the grinding status
# Input Parameters:
#   - path: The path to the audiosignal to analyse, 
#   - sr: The wanted sample_rate. The Audiosignal gets automaticly downsampled to this sample_rate. Default: 8000. 
def analyseAudio(data, sr):
    variance = np.std(data, ddof=1) ** 2

    if 0:
        variance = 1.0
        data = data / np.std(data, ddof=1)

    n = len(data)
    dt = 1 / sr
    pad = 1  # pad the time series with zeroes (recommended)

    # These parameters control the filter which is used für the wavelet funcion and controls the area of frequencies that are analysed
    # For this extraction a red-noise-filter is used, this is defined in cwtUtilsGrindingStatus line 355
    dj = 0.02
    s0 = 7.5 * dt
    j1 = 3 / dj
    lag1 = 0.95
    noice_multiplier = 1.2
    mother = 'MORLET'

    # Wavelet transform:
    wave, rpms, scale, coi = wavelet(data, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum

    # Significance levels:
    signif = wave_signif(variance * noice_multiplier, dt=dt, sigtest=0, scale=scale,
        lag1=lag1, mother=mother)
    
    # expand signif --> (J+1)x(N) array
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
    sig95 = power / sig95

    return sig95, rpms

# Input Parameters:
#   - path: The path to the audiosignal to analyse, 
#   - sr: The wanted sample_rate. The Audiosignal gets automaticly downsampled to this sample_rate.
#   - precisonMultiplier: Defines in how many parts one second of the audio signal gets splitted.
def getMostIntenseRelevantFrequencies(data, sr, precisonMultiplier):
    sig95, rpms = analyseAudio(data, sr)

    length = len(sig95[0])
    seconds = int(np.floor(length/sr))

    frequencyMessured = []

    top = int(sr/precisonMultiplier)
    bottom = 0
    
    for s in range(0, seconds * precisonMultiplier):
        maxIntensity = 0
        maxIndex = 0
        for frequency in range(0, len(sig95)):
            current = np.max(sig95[frequency][bottom:top])
            if current > maxIntensity:
                maxIntensity = current
                maxIndex = frequency

        frequencyMessured.append(rpms[maxIndex])

        top += int(sr/precisonMultiplier)
        bottom += int(sr/precisonMultiplier)

    return frequencyMessured

# Input Parameters:
#   - path: The path to the audiosignal to analyse, 
#   - sr: The wanted sample_rate. The Audiosignal gets automaticly downsampled to this sample_rate. Default: 8000. 
#   - precisonMultiplier: Defines in how many parts one second of the audio signal gets splitted. Default: 10.
#   - overSeconds: Defines the interval in which the grinding status is checked each.
#   - limit: Defines till which std the part is classified as not grinding
# Returns a array of the grinding status. True=Grinding. Array lenght equals data length in seconds / overSeconds.
def calculateGrindingStatus(data, sr=8000, precisonMultiplier=50, overSeconds=1, limit=15000):

    frequencies = getMostIntenseRelevantFrequencies(data=data, sr=sr, precisonMultiplier=precisonMultiplier)

    countOfPartsToBeChecked = int(len(frequencies) / precisonMultiplier / overSeconds)

    result = []
    stds = []

    for s in range(0, countOfPartsToBeChecked):
        rpmsForThisSecond = frequencies[s*precisonMultiplier*overSeconds:(s+1)*precisonMultiplier*overSeconds]
        stdOfSecond = np.std(rpmsForThisSecond)

        stds.append(stdOfSecond)

    filter = [1,1,1,1]
    stds = np.convolve(stds, filter, mode="same")

    for std in stds:
        if std > limit:
            result.append(True)
        else:
            result.append(False)
            
    return result