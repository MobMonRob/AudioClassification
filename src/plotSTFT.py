import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
import librosa


import numpy as np

PATH = "audio/16-11-08.wav"

audio, sr = librosa.load(PATH, sr = 8000)



N = len(audio)
T_x = 1 / sr
t_x = np.arange(N) * T_x
dr = 1 / T_x

g_std = sr/100
w = gaussian(4*sr, std=g_std, sym=True)  # symmetric Gaussian window

SFT = signal.ShortTimeFFT(w, int(sr/10), dr, mfft=32000, scale_to='magnitude', fft_mode='onesided')

Sx = SFT.stft(audio)

print(Sx.shape)

fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
ax1.set_title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Gaussian window, " +
              rf"$\sigma_t={g_std*SFT.T}\,$s)")
ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
               rf"$\Delta t = {SFT.delta_t:g}\,$s)",
        ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
               rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
        xlim=(t_lo, t_hi), ylim=(0, 500))

im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
                 extent=SFT.extent(N), cmap='viridis')

fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

# Shade areas where window slices stick out to the side:
for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
                 (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
    ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)
for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line:
    ax1.axvline(t_, color='y', linestyle='--', alpha=0.5)

ax1.legend()
fig1.tight_layout()
plt.show()
