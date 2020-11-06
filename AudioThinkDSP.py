# ------------------------------------------
# Librerias utilizadas
# ------------------------------------------
from thinkdsp import read_wave
import matplotlib.pyplot as plt
import os

# ------------------------------------------
# Lectura de los archivos
# ------------------------------------------
track1 = read_wave("base2_clean.wav")
track2 = read_wave("base4_overdrive-0.wav")

# ------------------------------------------
# Grafica de comparativa temporal de las dos
# señales otorgadas
# ------------------------------------------
#fig, ax0 = plt.subplots(figsize=(10,6))
#track1.plot()
#track2.plot()
#ax0.set_xlabel("Timepo [s]")
#ax0.set_ylabel("Amplitud")
#ax0.set_title("Tracks")
#ax0.grid(True)

# ------------------------------------------
# Obtencion de los espectros y filtrado
# ------------------------------------------
espectro1 = track1.make_spectrum()
espectroFil = track1.make_spectrum()
print(id(espectroFil))
print(id(espectro1))
# Recorte de espectro
espectroFil.low_pass(3000)
# Filtrado adicional
espectroFil.band_stop(50,350,0.7)
espectroFil.band_stop(110,140,0.3)
espectro2 = track2.make_spectrum()

# ------------------------------------------
# Grafica de comparativa frecuencial de los
# espectros de la señal filtrada y la señal
# modificada
# ------------------------------------------
fig, ax1 = plt.subplots(figsize=(10,6))
espectro1.plot(high=5000)
espectroFil.plot(high=5000)
espectro2.plot(high=5000)
ax1.set_xlabel("Frecuencia [Hz]")
ax1.set_ylabel("Amplitud")
ax1.set_title("Espectro de ambos tracks")
ax1.grid(True)

# ------------------------------------------
# Se hace la IFFT y se grafica para observar
# las diferencias
# ------------------------------------------
trackFil = espectroFil.make_wave()
fig, ax2 = plt.subplots(figsize=(10,6))
trackFil.plot()
track2.plot()
ax2.set_xlabel("Timepo [s]")
ax2.set_ylabel("Amplitud")
ax2.set_title("Tracks")
ax2.grid(True)

plt.show()

# ------------------------------------------
# Se genera el track filtrado para escucharlo y
# compararlo con el trac modificado
# ------------------------------------------

if os.path.exists("base_filtrada.wav"):
  print("The file does exist")
  os.remove("base_filtrada.wav")
else:
  print("The file does not exist")

trackFil.write("base_filtrada.wav")