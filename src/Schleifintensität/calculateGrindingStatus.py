import matplotlib.pylab as plt
import librosa
import numpy as np

from waveletFunctionsSchleifen import wave_signif, wavelet

# Uses a Wavelet transform and a significance test to extract the relevant frequencies for the grinding status
# Input Parameters:
#   - path: The path to the audiosignal to analyse, 
#   - sr: The wanted sample_rate. The Audiosignal gets automaticly downsampled to this sample_rate. Default: 8000. 
def analyseAudio(path, sr):
    # READ THE DATA
    sst, sample_rate = librosa.load(path, sr=sr)
    variance = np.std(sst, ddof=1) ** 2

    if 0:
        variance = 1.0
        sst = sst / np.std(sst, ddof=1)

    n = len(sst)
    dt = 1 / sample_rate
    pad = 1  # pad the time series with zeroes (recommended)

    #These parameters control the filter which is used für the wavelet funcion and controls the area of frequencies that are analysed
    dj = 0.02
    s0 = 7.5 * dt
    j1 = 3 / dj
    lag1 = 0.95
    noice_multiplier = 1.2
    mother = 'MORLET'

    # Wavelet transform:
    wave, rpms, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
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
def getMostIntenseRelevantFrequencies(path, sr, precisonMultiplier):
    sig95, rpms = analyseAudio(path, sr)

    length = len(sig95[0])
    seconds = int(np.floor(length/sr))

    intensitys = []
    frequencyMessured = []

    top = int(sr/precisonMultiplier)
    bottom = 0
    
    for s in range(0, seconds * precisonMultiplier):
        intensity = 0
        maxIndex = 0
        for frequency in range(0, len(sig95)):
            current = np.max(sig95[frequency][bottom:top])
            if current > intensity:
                intensity = current
                maxIndex = frequency

        intensitys.append(intensity * 10000)
        frequencyMessured.append(rpms[maxIndex])

        top += int(sr/precisonMultiplier)
        bottom += int(sr/precisonMultiplier)

    return frequencyMessured, intensitys

# Input Parameters:
#   - path: The path to the audiosignal to analyse, 
#   - sr: The wanted sample_rate. The Audiosignal gets automaticly downsampled to this sample_rate. Default: 8000. 
#   - precisonMultiplier: Defines in how many parts one second of the audio signal gets splitted. Default: 10.
#   - overSeconds: Defines the interval in which the grinding status is checked each.
#   - limit: Defines till which std the part is classified as not grinding
def calculateVarianzAndClassify(path, sr=8000, precisonMultiplier=50, overSeconds=1, limit=7500):

    frequencies, intensitys = getMostIntenseRelevantFrequencies(path=path, sr=sr, precisonMultiplier=precisonMultiplier)

    countOfPartsToBeChecked = int(len(frequencies) / precisonMultiplier / overSeconds)

    result = []
    stds = []

    for s in range(0, countOfPartsToBeChecked):
        rpmsForThisSecond = frequencies[s*precisonMultiplier*overSeconds:(s+1)*precisonMultiplier*overSeconds]
        stdOfSecond = np.std(rpmsForThisSecond)

        stds.append(stdOfSecond)

    filter = [0.25, 0.25, 0.25, 0.25]
    stds_filter = np.convolve(stds, filter, mode="same")

    for std in stds_filter:
        if std > limit:
            result.append(1)
        else:
            result.append(0)
            
    return result, frequencies, stds, stds_filter

result, freqs, stds, std_filter = calculateVarianzAndClassify("audio/16-11-08.wav", sr=8000, precisonMultiplier=50, overSeconds=1)

fig = plt.figure(figsize=(9, 10))

plt.plot(result)
plt.xlabel('Time')
plt.ylabel('Is_Grinding')
plt.title('6000 - Air - 50 Scans pro Sekunde')
plt.ylim(-0.25,1.25)
plt.show()

