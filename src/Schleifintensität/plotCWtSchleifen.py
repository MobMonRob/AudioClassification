import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

import librosa

import numpy as np

from waveletFunctionsSchleifen import wave_signif, wavelet

def plotCWT(PATH, sr = 8000, start = 0, end = None):
    # READ THE DATA
    sst, sample_rate = librosa.load(PATH, sr=sr)

    start = sample_rate * start
    sst = sst[start:]

    if end is not None:
        end = sample_rate * end
        sst = sst[:end]

    variance = np.std(sst, ddof=1) ** 2

    if 0:
        variance = 1.0
        sst = sst / np.std(sst, ddof=1)
    n = len(sst)
    dt = 1 / sample_rate
    time = np.arange(len(sst)) * dt + 0  # construct time array
    pad = 1  # pad the time series with zeroes (recommended)
    dj = 0.02  # this will do 4 sub-octaves per octave
    s0 = 7.5 * dt  # this says start at a scale of 6 months
    j1 = 3 / dj  # this says do 7 powers-of-two with dj sub-octaves each
    lag1 = 0.95  # lag-1 autocorrelation for red noise background
    noice_multiplier = 1.2
    mother = 'MORLET'


    # Wavelet transform:
    wave, rpms, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
    global_ws = (np.sum(power, axis=1) / n)  # time-average over all times

    # Significance levels:
    signif = wave_signif(variance * noice_multiplier, dt=dt, sigtest=0, scale=scale,
        lag1=lag1, mother=mother)
    # expand signif --> (J+1)x(N) array
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
    sig95 = power / sig95  # where ratio > 1, power is significant

    # Global wavelet spectrum & significance levels:
    dof = n - scale  # the -scale corrects for padding at edges
    global_signif = wave_signif(variance * noice_multiplier, dt=dt, scale=scale, sigtest=1,
        lag1=lag1, dof=dof, mother=mother)

    # ------------------------------------------------------ Plotting

    max = int(n / 2**16) + 1
    delta_time = 2**16 * dt

    # --- Plot time series
    fig = plt.figure(figsize=(9, 10))
    gs = GridSpec(3, 2* max + 2, hspace=0.4, wspace=0.75)

    for i in range(0, max):

        plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95,
                            wspace=0, hspace=0)
        plt.subplot(gs[0, 2*i:2*(i+1)])
        plt.plot(time, sst, 'k')
        plt.xlim((i * delta_time, (i+1) * delta_time))

        if (i == 0):
            plt.xlabel('Time (seconds)')
            plt.ylabel('RPM')
            plt.title('6000 - Grinding')

        # --- Contour plot wavelet power spectrum
        splot = plt.subplot(gs[1, 2*i:2*(i+1)])
        levels = [0, 0.0625, 0.125, 0.25, 0.5, 999]
        # *** or use 'contour'

        CS = plt.contourf(time, rpms, power, len(levels))
        im = plt.contourf(CS, levels=levels,
            colors=['white', 'bisque', 'orange', 'orangered', 'darkred'])
        
        if (i == 0):
            plt.xlabel('Time (seconds)')
            plt.ylabel('RPM')
            plt.title('6000 - Air')

        plt.xlim((i * delta_time, (i+1) * delta_time))
        # 95# significance contour, levels at -99 (fake) and 1 (95# signif)
        plt.contour(time, rpms, sig95, [-99, 1], colors='k')

        # cone-of-influence, anything "below" is dubious
        plt.fill_between(time, coi * 0 + rpms[-1], coi, facecolor="none",
            edgecolor="#00000040", hatch='x')
        plt.plot(time, coi, 'k')
        # format y-scale
        splot.set_yscale('log', base=2)
        plt.ylim([np.min(rpms), np.max(rpms)])
        ax = plt.gca().yaxis
        ax.set_major_formatter(ticker.ScalarFormatter())
        splot.ticklabel_format(axis='y', style='plain')
        # set up the size and location of the colorbar
        # position=fig.add_axes([0.5,0.36,0.2,0.01])
        # plt.colorbar(im, cax=position, orientation='horizontal')
        #   , fraction=0.05, pad=0.5)

        # plt.subplots_adjust(right=0.7, top=0.9)

    # --- Plot global wavelet spectrum
    splot = plt.subplot(gs[1, -1])
    plt.plot(global_ws, rpms)
    plt.plot(global_signif, rpms, '--')
    plt.xlabel('Power')
    plt.ylabel('RPM')
    plt.title('GWS')
    plt.xlim([0, 1.25 * np.max(global_ws)])
    # format y-scale
    splot.set_yscale('log', base=2)
    plt.ylim([np.min(rpms), np.max(rpms)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ticker.ScalarFormatter())
    splot.ticklabel_format(axis='y', style='plain')

    plt.show()


def main():
    # plotCWT("audio/07-05-07.wav",)
    plotCWT("audio/16-11-08.wav", start=0, end=40)

    # analyseAudio("audio/07-05-02.wav", 8000)
    # analyseAudio("audio/07-05-03.wav", 8000)
    # analyseAudio("audio/07-05-04.wav", 8000)
    # analyseAudio("audio/07-05-05.wav", 8000)

if __name__ == '__main__':
    main()
