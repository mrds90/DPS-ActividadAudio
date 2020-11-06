# -*- coding: utf-8 -*-

"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Filtrado DIgital:
    En el siguiente script se ejemplifica el proceso de carga de filtros 
    digitales creados con la herramienta pyFDA y el uso de los mismos para el 
    filtrado de señales.

Autor Original: Albano Peñalva
Fecha: Abril 2020
Edición: Marcos Dominguez

Se adapta el codigo para hacer el filtrado de una señal de una pista de audio.

"""

# Librerías
from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import funciones_fft
from time import time
import os
plt.close('all') # cerrar gráficas anteriores

######################## Lectura del archivo de audio ########################

filename = 'base2_clean.wav'     # nombre de archivo
fs, data = wavfile.read(filename)   # frecuencia de muestreo y datos de la señal

# Definición de parámetro temporales
ts = 1 / fs                     # tiempo de muestreo
print('Frecuencia de muestreo: ',fs)
N = len(data)                   # número de muestras en el archivo de audio
t = np.linspace(0, N * ts, N)   # vector de tiempo
senial = data             # se extrae un canal de la pista de audio (el audio es estereo)
senial = senial * 3.3 / 2**16   # se escala la señal a voltios (considerando un CAD de 16bits y Vref 3.3V)

####################### Graficación señal temporal ###########################

# Se crea una gráfica 
fig1, ax1 = plt.subplots(2, 1, figsize=(15, 15), sharex=True)
fig1.suptitle('Tiempo', fontsize=18)

# Se grafica la señal
ax1[0].plot(t, senial, label='Señal a Filtrar')
ax1[0].set_ylabel('Tensión [V]', fontsize=15)
ax1[0].grid()
ax1[0].legend(loc="upper right", fontsize=15)
ax1[0].set_title('Filtrado FIR', fontsize=15)
ax1[1].plot(t, senial, label='Señal a Filtrar')
ax1[1].set_ylabel('Tensión [V]', fontsize=15)
ax1[1].grid()
ax1[1].legend(loc="upper right", fontsize=15)
ax1[1].set_xlabel('Tiempo [s]', fontsize=15)
ax1[1].set_xlim([0, ts*N])
ax1[1].set_title('Filtrado IIR', fontsize=15)

############ Cálculo y Graficación de la Transformada de Fourier #############

# Se calcula el espectro de la señal contaminada
f, senial_fft_mod = funciones_fft.fft_mag(senial, fs)

# Se crea una gráfica 
fig2, ax2 = plt.subplots(2, 1, figsize=(15, 15), sharex=True)
fig2.suptitle('Nombre Grafica', fontsize=18)

# Se grafica la magnitud del espectro (normalizado)
ax2[0].plot(f, senial_fft_mod/np.max(senial_fft_mod), label='Señal Contaminada')
ax2[0].set_ylabel('Magnitud (normalizada))', fontsize=15)
ax2[0].grid()
ax2[0].legend(loc="upper right", fontsize=15)
ax2[1].plot(f, senial_fft_mod/np.max(senial_fft_mod), label='Señal Contaminada')
ax2[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2[1].grid()
ax2[1].legend(loc="upper right", fontsize=15)
ax2[1].set_ylabel('Magnitud (normalizada))', fontsize=15)
ax2[1].set_xlim([0, 5000])

######################### Carga y Análisis de Filtros ########################

#############PASA ALTOS################
# Se cargan los archivo generado mediante pyFDA
filtro_fir_highPass = np.load('highPass1FIR.npz')
filtro_iir_highPass = np.load('highPass1IIR.npz') 
# Se extraen los coeficientes de numerador y denominador
Num_fir_highPass, Den_fir_highPass = filtro_fir_highPass['ba']     
Num_iir_highPass, Den_iir_highPass = filtro_iir_highPass['ba'] 
# Se calcula la respuesta en frecuencia de los filtros
f_fir_highPass, h_fir_highPass = signal.freqz(Num_fir_highPass, Den_fir_highPass, worN=f, fs=fs)
f_iir_highPass, h_iir_highPass = signal.freqz(Num_iir_highPass, Den_iir_highPass, worN=f, fs=fs)
# Se grafican las respuestas de los filtros
ax2[0].plot(f_fir_highPass, abs(h_fir_highPass), label='Filtro FIR', color='orange')
ax2[0].legend(loc="upper right", fontsize=15)
ax2[1].plot(f_iir_highPass, abs(h_iir_highPass),  label='Filtro IIR', color='green')
ax2[1].legend(loc="upper right", fontsize=15)
#############BANDSTOP################
# Se cargan los archivo generado mediante pyFDA
filtro_fir_bandStop1 = np.load('bandStop1FIR.npz')
filtro_iir_bandStop1 = np.load('bandStop1IIR.npz') 
# Se extraen los coeficientes de numerador y denominador
Num_fir_bandStop1, Den_fir_bandStop1 = filtro_fir_bandStop1['ba']     
Num_iir_bandStop1, Den_iir_bandStop1 = filtro_iir_bandStop1['ba'] 
# Se calcula la respuesta en frecuencia de los filtros
f_fir_bandStop1, h_fir_bandStop1 = signal.freqz(Num_fir_bandStop1, Den_fir_bandStop1, worN=f, fs=fs)
f_iir_bandStop1, h_iir_bandStop1 = signal.freqz(Num_iir_bandStop1, Den_iir_bandStop1, worN=f, fs=fs)
# Se grafican las respuestas de los filtros
ax2[0].plot(f_fir_bandStop1, abs(h_fir_bandStop1), color='orange')
ax2[0].legend(loc="upper right", fontsize=15)
ax2[1].plot(f_iir_bandStop1, abs(h_iir_bandStop1),  color='green')
ax2[1].legend(loc="upper right", fontsize=15)
#############BANDSTOP2################
# Se cargan los archivo generado mediante pyFDA
filtro_fir_bandStop2 = np.load('bandStop2FIR.npz')
filtro_iir_bandStop2 = np.load('bandStop2IIR.npz') 
# Se extraen los coeficientes de numerador y denominador
Num_fir_bandStop2, Den_fir_bandStop2 = filtro_fir_bandStop2['ba']     
Num_iir_bandStop2, Den_iir_bandStop2 = filtro_iir_bandStop2['ba'] 
# Se calcula la respuesta en frecuencia de los filtros
f_fir_bandStop2, h_fir_bandStop2 = signal.freqz(Num_fir_bandStop2, Den_fir_bandStop2, worN=f, fs=fs)
f_iir_bandStop2, h_iir_bandStop2 = signal.freqz(Num_iir_bandStop2, Den_iir_bandStop2, worN=f, fs=fs)
# Se grafican las respuestas de los filtros
ax2[0].plot(f_fir_bandStop2, abs(h_fir_bandStop2), color='orange')
ax2[0].legend(loc="upper right", fontsize=15)
ax2[1].plot(f_iir_bandStop2, abs(h_iir_bandStop2),  color='green')
ax2[1].legend(loc="upper right", fontsize=15)
#############BANDSTOP3################
# Se cargan los archivo generado mediante pyFDA
filtro_fir_bandStop3 = np.load('bandStop3FIR.npz')
filtro_iir_bandStop3 = np.load('bandStop3IIR.npz') 
# Se extraen los coeficientes de numerador y denominador
Num_fir_bandStop3, Den_fir_bandStop3 = filtro_fir_bandStop3['ba']     
Num_iir_bandStop3, Den_iir_bandStop3 = filtro_iir_bandStop3['ba'] 
# Se calcula la respuesta en frecuencia de los filtros
f_fir_bandStop3, h_fir_bandStop3 = signal.freqz(Num_fir_bandStop3, Den_fir_bandStop3, worN=f, fs=fs)
f_iir_bandStop3, h_iir_bandStop3 = signal.freqz(Num_iir_bandStop3, Den_iir_bandStop3, worN=f, fs=fs)
# Se grafican las respuestas de los filtros
ax2[0].plot(f_fir_bandStop3, abs(h_fir_bandStop3), color='orange')
ax2[0].legend(loc="upper right", fontsize=15)
ax2[1].plot(f_iir_bandStop3, abs(h_iir_bandStop3),  color='green')
ax2[1].legend(loc="upper right", fontsize=15)

#############LOWPASS################
# Se cargan los archivo generado mediante pyFDA
filtro_fir_lowPass1 = np.load('lowPass1FIR.npz')
filtro_iir_lowPass1 = np.load('lowPass1IIR.npz') 
# Se extraen los coeficientes de numerador y denominador
Num_fir_lowPass1, Den_fir_lowPass1 = filtro_fir_lowPass1['ba']     
Num_iir_lowPass1, Den_iir_lowPass1 = filtro_iir_lowPass1['ba'] 
# Se calcula la respuesta en frecuencia de los filtros
f_fir_lowPass1, h_fir_lowPass1 = signal.freqz(Num_fir_lowPass1, Den_fir_lowPass1, worN=f, fs=fs)
f_iir_lowPass1, h_iir_lowPass1 = signal.freqz(Num_iir_lowPass1, Den_iir_lowPass1, worN=f, fs=fs)
# Se grafican las respuestas de los filtros
ax2[0].plot(f_fir_lowPass1, abs(h_fir_lowPass1), color='orange')
ax2[0].legend(loc="upper right", fontsize=15)
ax2[1].plot(f_iir_lowPass1, abs(h_iir_lowPass1),  color='green')
ax2[1].legend(loc="upper right", fontsize=15)


############################ Filtrado de la Señal ############################

# Se aplica el filtrado sobre la señal
#FIR
start=time()
senial_fir = signal.lfilter(Num_fir_highPass, Den_fir_highPass, senial)
senial_fir = signal.lfilter(Num_fir_bandStop1, Den_fir_bandStop1, senial_fir)
senial_fir = signal.lfilter(Num_fir_bandStop2, Den_fir_bandStop2, senial_fir)
senial_fir = signal.lfilter(Num_fir_bandStop3, Den_fir_bandStop3, senial_fir)
senial_fir = signal.lfilter(Num_fir_lowPass1, Den_fir_lowPass1, senial_fir)
senial_fir=2.8*senial_fir
print('tiempo de procesamiento FIR: ', time()-start)
#IIR
start=time()
senial_iir = signal.lfilter(Num_iir_highPass, Den_iir_highPass, senial)
senial_iir = signal.lfilter(Num_iir_bandStop1, Den_iir_bandStop1, senial_iir)
senial_iir = signal.lfilter(Num_iir_bandStop2, Den_iir_bandStop2, senial_iir)
senial_iir = signal.lfilter(Num_iir_bandStop3, Den_iir_bandStop3, senial_iir)
senial_iir = signal.lfilter(Num_iir_lowPass1, Den_iir_lowPass1, senial_iir)
senial_iir=2.8*senial_iir
print('tiempo de procesamiento IIR: ', time()-start)

# Se grafican las señales filtradas
ax1[0].plot(t, senial_fir, label='Señal Filtrada (FIR)', color='red')
ax1[0].legend(loc="upper right", fontsize=15)
ax1[1].plot(t, senial_iir, label='Señal Filtrada (IIR)', color='purple')
ax1[1].legend(loc="upper right", fontsize=15)

# Se calculan y grafican sus espectros (normalizados)
f1_fir, senial_fir_fft_mod = funciones_fft.fft_mag(senial_fir, fs)
f1_iir, senial_iir_fft_mod = funciones_fft.fft_mag(senial_iir, fs)
ax2[0].plot(f1_fir, senial_fir_fft_mod/np.max(senial_fft_mod), label='Senial Filtrada FIR', color='red')
ax2[0].legend(loc="upper right", fontsize=15)
ax2[1].plot(f1_iir, senial_iir_fft_mod/np.max(senial_fft_mod), label='Senial Filtrada IIR', color='purple')
ax2[1].legend(loc="upper right", fontsize=15)


plt.show()
# crear archivo wav
if os.path.exists("base_filtrada_FIR.wav"):
  print("The file does exist")
  os.remove("base_filtrada_FIR.wav")
else:
  print("The file does not exist")

wavfile.write('base_filtrada_FIR.wav',fs, senial_fir)


if os.path.exists("base_filtrada_IIR.wav"):
  print("The file does exist")
  os.remove("base_filtrada_IIR.wav")
else:
  print("The file does not exist")

wavfile.write('base_filtrada_IIR.wav',fs, senial_iir)