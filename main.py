import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import pywt.data


def read(filepath):
    with open(filepath, 'r') as fp:
        text = fp.read()
        return text


dane = read('D:\Projekty\INŻYNIERKA\Dane\Sin_zmienne.txt')
a = np.genfromtxt(StringIO(dane), skip_header=0)
# T- ilość danych dla jednego przedziału  <-- ZAMIENIC PÓŹNIEJ NA CZAS np. 24h
# (czas podzielimy przez krok próbkowania by otrzymać ilość danych
T = 1000
t = np.arange(T)
b = a.reshape(a.size//T, -1)

print(str(b.shape))
print(b[2, 1])

# Signal

plt.figure(figsize=(10, 6), facecolor='gray')

plt.subplot(2, 1, 1, facecolor='silver')
plt.title('Signal', fontsize=20)
plt.plot(t, b[0], linewidth=1, label='b[0]')
plt.plot(t, b[1], linewidth=1.2, label='b[1]')
plt.plot(t, b[2], linewidth=1, label='b[2]')

plt.xlabel('Time', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.legend(loc='upper right')

plt.axhline(y=0, xmin=0, xmax=1000, color='black', linewidth=0.3)
plt.axvline(x=0, ymin=0, ymax=1000, color='black', linewidth=0.3)

# SPECTRAL ANALYSIS

freq = np.fft.rfftfreq(t.shape[-1])
widmo2 = np.fft.rfft(b[2, :])/T
widmo0 = np.fft.rfft(b[0, :])/T
widmo1 = np.fft.rfft(b[1, :])/T

plt.subplot(2, 1, 2, facecolor='silver')
plt.title('FFT', fontsize=20)
image = plt.imread('D:\Projekty\INŻYNIERKA\Dane\Screenshot_1.png')
# Widmo na wykresie z Shearera
# plt.imshow(image, extent=[freq.min(), freq.max(), -2, 1])
plt.plot(freq, widmo0, label='widmo[0]')
plt.plot(freq, widmo1, label='widmo[1]')
plt.plot(freq, widmo2, label='widmo[2]')

plt.xlabel('Frequency', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.legend(loc='upper right')

plt.axhline(y=0, xmin=0, xmax=1, color='black', linewidth=0.3)
plt.axvline(x=0, ymin=-2500, ymax=6500, color='black', linewidth=0.3)

plt.tight_layout()
plt.show()

# CONTINUOUS WAVELET TRANSFORM

dekomp, freq = pywt.cwt(b[0], np.arange(1, 101, 0.4), 'gaus1')
dekomp1, freq1 = pywt.cwt(b[1], np.arange(1, 101, 0.4), 'gaus1')
dekomp2, freq2 = pywt.cwt(b[2], np.arange(1, 101, 0.4), 'gaus1')

fig2 = plt.figure(figsize=(12, 8), facecolor='gray')
plt.subplot(2, 1, 1)
plt.title('Spectral decomposition', fontsize=20)
plt.imshow(dekomp, cmap='hsv')
plt.colorbar(alpha=1)

plt.subplot(2, 1, 2)
plt.imshow(dekomp1, cmap='hsv', alpha=1)
plt.colorbar(alpha=0.9)

# plt.tight_layout()
plt.show()
