import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import time

import sys, os

sys.path.append(os.path.realpath('../basicFunctions/'))
sys.path.append(os.path.realpath('../basicFunctions_C/'))
import smsF0DetectionTwm as fd
import smsWavplayer as wp
import smsPeakProcessing as PP

try:
  import basicFunctions_C as GS
except ImportError:
  import smsGenSpecSines as GS
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  print "NOTE: Cython modules for some functions were not imported, the processing will be slow"
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

def hpsModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf) :
  # Analysis/synthesis of a sound using the harmonic plus stochastic model
  # x: input sound, fs: sampling rate, w: analysis window (odd size), 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # maxhd: max. relative deviation in harmonic detection (ex: .2)
  # stocf: decimation factor of mag spectrum for stochastic analysis
  # y: output sound, yh: harmonic component, ys: stochastic component
  
  x = np.float32(x) / (2**15)                                   # normalize input signal

  fig = plt.figure(figsize = (10.5, 6.5), dpi = 100)
  ax1 = plt.subplot2grid((4, 6), (0, 1), colspan = 4)
  ax1.set_position([0.10, 0.77, 0.8, 0.16])
  ax1.set_xlim(0, 10000)
  ax1.set_ylim(x.min(), x.max())
  ax1.set_title("Input Signal")
  plt.setp(ax1.get_xticklabels(), visible = False)
  
  ax2 = plt.subplot2grid((4, 6), (1, 1), colspan = 4, sharex = ax1, sharey = ax1)
  ax2.set_position([0.10, 0.55, 0.8, 0.16])
  ax2.set_xlim(0, 10000)
  ax2.set_ylim(x.min(), x.max())
  ax2.set_title("Output Signal")

  ax3 = plt.subplot2grid((4, 6), (2, 0), rowspan = 2, colspan = 2)
  ax3.set_position([0.05, 0.08, 0.35, 0.35])
  ax3.set_title("Frame")
  ax3.set_xlim(0, w.size)
  
  # ax4 = plt.subplot2grid((4, 4), (2, 1), rowspan = 2)
  # plt.title("Windowed")

  ax5 = plt.subplot2grid((4, 6), (2, 3), rowspan = 2, colspan = 4)
  ax5.set_position([0.47, 0.08, 0.5, 0.35])
  ax5.set_title("Spectrum")
  ax5.set_xlabel("Frequency (Hz)")
  ax5.set_ylabel("Amplitude (dB)")
  ax5.set_xlim(0, fs/2)

  hN = N/2                                                      # size of positive spectrum
  hM = (w.size+1)/2                                             # half analysis window size
  Ns = 512                                                      # FFT size for synthesis (even)
  H = Ns/4                                                      # Hop size used for analysis and synthesis
  hNs = Ns/2      
  pin = max(hNs, hM)                                            # initialize sound pointer in middle of analysis window          
  pend = x.size - max(hNs, hM)                                  # last sample to start a frame
  fftbuffer = np.zeros(N)                                       # initialize buffer for FFT
  yhw = np.zeros(Ns)                                            # initialize output sound frame
  ysw = np.zeros(Ns)                                            # initialize output sound frame
  yh = np.zeros(x.size)                                         # initialize output array
  ys = np.zeros(x.size)                                         # initialize output array
  w = w / sum(w)                                                # normalize analysis window
  sw = np.zeros(Ns)     
  ow = triang(2*H)                                              # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                       # synthesis window
  bh = bh / sum(bh)                                             # normalize synthesis window
  wr = bh                                                       # window for residual
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
  sws = H*hanning(Ns)/2                                         # synthesis window for stochastic
  
  ax1.plot(x[:10000])
  plt.draw()

  while pin<pend:       
    
    rect = patches.Rectangle((pin-hM, -2**7), width = w.size, height = 2**15, color = 'red', alpha = 0.3)
    ax1.add_patch(rect)  
    plt.draw()
    rect.remove()
  
  #-----analysis-----             
    ax3.cla()
    ax3.set_title("Frame")
    ax3.plot(x[pin-hM:pin+hM-1])
    ax3.set_xlim(0, w.size)
    ax3.ticklabel_format(scilimits = (-3,3))                     # use scientific limits above 1e3
    plt.draw()
    ax3.set_ylim(ax3.get_ylim())
    ax3.plot(w, 'r')
    plt.draw()
    
    xw = x[pin-hM:pin+hM-1] * w                                  # window the input sound
    ax3.cla()
    ax3.set_title("Windowed Frame")
    ax3.plot(xw, 'b')
    ax3.set_xlim(0, w.size)
    ax3.ticklabel_format(scilimits = (-3,3))                     # use scientific limits above 1e3
    plt.draw()

    fftbuffer = np.zeros(N)                                      # reset buffer
    fftbuffer[:hM] = xw[hM-1:]                                   # zero-phase window in fftbuffer
    fftbuffer[N-hM+1:] = xw[:hM-1]              
    ax3.cla()
    ax3.set_title("Windowed Frame zero-phase")
    ax3.plot(fftbuffer, 'b')
    ax3.set_xlim(0, w.size)
    ax3.ticklabel_format(scilimits = (-3,3))                     # use scientific limits above 1e3
    plt.draw()


    X = fft(fftbuffer)                                           # compute FFT
    mX = 20 * np.log10( abs(X[:hN]) )                            # magnitude spectrum of positive frequencies
    ploc = PP.peakDetection(mX, hN, t)                 
    pX = np.unwrap( np.angle(X[:hN]) )                           # unwrapped phase spect. of positive freq.    
    iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)            # refine peak values
    
    freq = np.arange(0, fs/2, fs/N)                              # frequency axis in Hz
    freq = freq[:freq.size-1]
    ax5.cla()
    ax5.set_title("Spectrum")
    ax5.set_xlabel("Frequency (Hz)")
    ax5.set_ylabel("Amplitude (dB)")
    ax5.set_xlim(0, fs/2)
    ax5.plot(freq, mX, 'b')
    ax5.set_ylim(ax5.get_ylim())
    ax5.fill_between(freq, ax5.get_ylim()[0], mX, facecolor = 'blue', alpha = 0.3)
    plt.draw()
    ax5.plot(np.float32(iploc)/N*fs, ipmag, 'ro', ms = 4, alpha = 0.4)
    plt.draw()  

    f0 = fd.f0DetectionTwm(iploc, ipmag, N, fs, f0et, minf0, maxf0)  # find f0
    
    if f0 > 0:
      loc = np.where(iploc/N*fs == f0)[0]
      if loc.size == 0: loc = np.argmin(np.abs(iploc/N*fs-f0))   # closest peak location
      ax5.plot(f0, ipmag[loc], 'go', ms = 4, alpha = 1)
      plt.draw()
    
    hloc = np.zeros(nH)                                          # initialize harmonic locations
    hmag = np.zeros(nH)-100                                      # initialize harmonic magnitudes
    hphase = np.zeros(nH)                                        # initialize harmonic phases
    hf = (f0>0)*(f0*np.arange(1, nH+1))                          # initialize harmonic frequencies
    hi = 0                                                       # initialize harmonic index
    npeaks = ploc.size                                           # number of peaks found

    while f0>0 and hi<nH and hf[hi]<fs/2 :                       # find harmonic peaks
      dev = min(abs(iploc/N*fs - hf[hi]))
      pei = np.argmin(abs(iploc/N*fs - hf[hi]))                  # closest peak
      if ( hi==0 or not any(hloc[:hi]==iploc[pei]) ) and dev<maxhd*hf[hi] :
        hloc[hi] = iploc[pei]                                    # harmonic locations
        hmag[hi] = ipmag[pei]                                    # harmonic magnitudes
        hphase[hi] = ipphase[pei]                                # harmonic phases
      hi += 1                                                    # increase harmonic index
    
    ax5.plot(np.float32(hloc)/N*fs, hmag, 'yo', ms = 4, alpha = 0.7)
    plt.draw()
    hloc = (hloc!=0) * (hloc*Ns/N)                               # synth. locs

    ri = pin-hNs-1                                               # input sound pointer for residual analysis
    xw2 = x[ri:ri+Ns]*wr                                         # window the input sound                                       
    fftbuffer = np.zeros(Ns)                                     # reset buffer
    fftbuffer[:hNs] = xw2[hNs:]                                  # zero-phase window in fftbuffer
    fftbuffer[hNs:] = xw2[:hNs]              
    X2 = fft(fftbuffer)                                          # compute FFT for residual analysis
  
  #-----synthesis-----
    Yh = GS.genSpecSines(hloc, hmag, hphase, Ns)                    # generate spec sines          
    Xr = X2-Yh                                                   # get the residual complex spectrum
    mXr = 20 * np.log10( abs(Xr[:hNs]) )                         # magnitude spectrum of residual
    mXrenv = resample(np.maximum(-200, mXr), mXr.size*stocf)     # decimate the magnitude spectrum

    mYs = resample(mXrenv, hNs)                                  # interpolate to original size
    pYs = 2*np.pi*np.random.rand(hNs)                            # generate phase random values

    Ys = np.zeros(Ns, dtype = complex)
    Ys[:hNs] = 10**(mYs/20) * np.exp(1j*pYs)                     # generate positive freq.
    Ys[hNs+1:] = 10**(mYs[:0:-1]/20) * np.exp(-1j*pYs[:0:-1])    # generate negative freq.

    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real( ifft(Yh) )                              # inverse FFT
    ax3.cla()
    ax3.set_title("Reconstructed Frame")
    ax3.plot(fftbuffer, 'g')
    ax3.set_xlim(0, w.size)
    ax3.ticklabel_format(scilimits = (-3,3))                     # use scientific limits above 1e3
    plt.draw()

    yhw[:hNs-1] = fftbuffer[hNs+1:]                              # undo zero-phase window
    yhw[hNs-1:] = fftbuffer[:hNs+1] 
    
    ax3.cla()
    ax3.set_title("Reconstructed Frame")
    ax3.plot(yhw, 'g')
    ax3.set_xlim(0, w.size)
    ax3.ticklabel_format(scilimits = (-3,3))                     # use scientific limits above 1e3
    plt.draw()

    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real( ifft(Ys) )
    ysw[:hNs-1] = fftbuffer[hNs+1:]                              # residual in time domain using inverse FFT
    ysw[hNs-1:] = fftbuffer[:hNs+1]

    yh[ri:ri+Ns] += sw*yhw                                       # overlap-add for sines
    ys[ri:ri+Ns] += sws*ysw                                      # overlap-add for stochastic
    pin += H                                                     # advance sound pointer
    
    ax3.cla()
    ax3.set_title("Reconstructed Frame")
    ax3.plot(sw*yhw, 'g')
    ax3.set_xlim(0, w.size)
    ax3.ticklabel_format(scilimits = (-3,3))                     # use scientific limits above 1e3
    plt.draw()

    rect2 = patches.Rectangle((pin-hM, -2**7), width = Ns, height = 2**15, color = 'green', alpha = 0.3)
    ax2.cla()
    ax2.set_xlim(0, 10000)
    ax2.set_ylim(x.min(), x.max())
    ax2.set_title("Output Signal")
    ax2.add_patch(rect2)  
    ax2.plot(yh, 'b')
    plt.draw()
    rect2.remove()

  y = yh+ys
  return y, yh, ys


(fs, x) = wp.wavread('speech-female.wav')
# wp.play(x, fs)

w = np.hamming(1025)
N = 1024
t = -120
nH = 30
minf0 = 200
maxf0 = 500
f0et = 5
maxhd = 0.2
stocf = 0.5
y, yh, ys = hpsModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf)

y *= 2**15
y = y.astype(np.int16)

yh *= 2**15
yh = yh.astype(np.int16)

ys *= 2**15
ys = ys.astype(np.int16)

wp.play(y, fs)
wp.play(yh, fs)
wp.play(ys, fs)
