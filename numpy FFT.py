import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
cwd=os.getcwd()
#importing CSV data file
path=cwd+'\\data.csv' #save data.csv at the same location of python file
#replace path of data file 
data=pd.read_csv(path)
t=data.t
acc=data.acc
#dt is time increment in every data
dt=0.01
#calculating length of data
n = len(acc)
#computing frequency using scipy fftpack
freq = np.fft.fftfreq(n, d=dt)
# Fast Fourier Transform of Acceleration
accfft = np.array(np.fft.fft(acc, axis=0))
#computing absolute value of fft
Accfft=np.abs(accfft)*dt
#plotting the spectral
plt.subplot(3,1,1)
plt.plot(data.t,data.acc,lw=0.5,color="k")
plt.xlabel("time (sec)")
plt.ylabel(' Amplitude')
plt.subplot(3,1,2)
plt.plot(freq,Accfft,lw=0.6, color='k')
plt.xlabel('frequency, Hz')
plt.ylabel(' Amplitude')
plt.subplot(3,1,3)
#Plot of FFT in log scale
plt.loglog(freq,Accfft,lw=0.6, color='k')
plt.xlabel('frequency, Hz')
plt.ylabel(' Amplitude')
plt.grid()
plt.tight_layout()