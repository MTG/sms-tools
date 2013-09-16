from PySide.QtCore import *
from PySide.QtGui import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift

import sys, os, functools

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions_C/'))

#sys.path.append(os.path.realpath('../basicFunctions/'))
#sys.path.append(os.path.realpath('../basicFunctions_C/'))
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

import smsGui

class MainWindow(QDialog, smsGui.Ui_MainWindow):

  def __init__(self, x, fs, parent = None):
    super(MainWindow, self).__init__(parent)
    self.setupUi(self)

    self.nextButton.clicked.connect(functools.partial(self.playButtonClicked, x, fs, 1, True, True))
    self.playButton.clicked.connect(functools.partial(self.playButtonClicked, x, fs, 1, True))
    self.rewindButton.clicked.connect(functools.partial(self.playButtonClicked, x, fs, 100, True))
    self.doubleRewindButton.clicked.connect(functools.partial(self.playButtonClicked, x, fs, 500, True))
    self.endButton.clicked.connect(functools.partial(self.playButtonClicked, x, fs, 0, False))

  def playYClicked(self, x, fs):
    wp.play(x, fs)

  def playYClicked(self, y, fs):
    wp.play(y, fs)

  def playYhClicked(self, yh, fs):
    wp.play(yh, fs)

  def playYsClicked(self, ys, fs):
    wp.play(ys, fs)

  # def pauseButtonClicked(self):
  #  	time.sleep(5) 										# pause execution for 5 seconds

  def playButtonClicked(self, x, fs, step, plot, process = False):
    
    w = np.hamming(int(self.w.text()))
    N = int(self.N.text())
    t = int(self.t.text())
    nH = int(self.nH.text())
    minf0 = int(self.minf0.text())
    maxf0 = int(self.maxf0.text())
    f0et = float(self.f0et.text())
    maxhd = float(self.maxhd.text())
    stocf = float(self.stocf.text())
    nFrameStart = int(self.nFrameStart.text())
    
    self.playX.setEnabled(False)
    self.playY.setEnabled(False)
    self.playYh.setEnabled(False)
    self.playYs.setEnabled(False)
    
    self.w.setEnabled(False)
    self.N.setEnabled(False)
    self.t.setEnabled(False)
    self.nH.setEnabled(False)
    self.minf0.setEnabled(False)
    self.maxf0.setEnabled(False)
    self.f0et.setEnabled(False)
    self.maxhd.setEnabled(False)
    self.stocf.setEnabled(False)

    y, yh, ys = self.hps(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, step, plot, nFrameStart, process)

    self.playX.setEnabled(True)
    self.playY.setEnabled(True)
    self.playYh.setEnabled(True)
    self.playYs.setEnabled(True)

    self.w.setEnabled(True)
    self.N.setEnabled(True)
    self.t.setEnabled(True)
    self.nH.setEnabled(True)
    self.minf0.setEnabled(True)
    self.maxf0.setEnabled(True)
    self.f0et.setEnabled(True)
    self.maxhd.setEnabled(True)
    self.stocf.setEnabled(True)

    self.playX.clicked.connect(functools.partial(self.playYClicked, x, fs))
    self.playY.clicked.connect(functools.partial(self.playYClicked, y, fs))
    self.playYh.clicked.connect(functools.partial(self.playYClicked, yh, fs))
    self.playYs.clicked.connect(functools.partial(self.playYClicked, ys, fs))

  def hps(self, x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, step, plot, nFrameStart, process):
    # Analysis/synthesis of a sound using the harmonic plus stochastic model
    # x: input sound, fs: sampling rate, w: analysis window (odd size), 
    # N: FFT size (minimum 512), t: threshold in negative dB, 
    # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
    # maxf0: maximim f0 frequency in Hz, 
    # f0et: error threshold in the f0 detection (ex: 5),
    # maxhd: max. relative deviation in harmonic detection (ex: .2)
    # stocf: decimation factor of mag spectrum for stochastic analysis
    # y: output sound, yh: harmonic component, ys: stochastic component

    freq_range = 10000 # fs/2 by default
    
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
    lastyhloc = np.zeros(nH)                                      # initialize synthesis harmonic locations
    yhphase = 2*np.pi * np.random.rand(nH)                        # initialize synthesis harmonic phases     

    n_frame = 0 																									# initialize number of frames counter
    
    if plot:
	    
	  #-----initialize plots-----
	    
	    plt.ion() 																									 # activate interactive mode 
	    clip_in = 0.0                                                # samples to clip input/output signal
	    clip_spec = 0.0                                              # number of frames to clip spectrogram
	    freq = np.arange(0, freq_range, fs/N)                        # frequency axis in Hz
	    freq = freq[:freq.size-1]
	    time = np.arange(0, np.float32(x.size)/fs, 1.0/fs)           # time axis in seconds
	    n_bins = freq.size 																					 # number of total bins in the freq_range
	    specgram = np.ones((n_bins, pend/H)) * -200                  # initialize spectrogram
	    prev_harmonics = np.zeros(nH-1)                              # previous harmonics to create harmonic trajectories
	    prev_f0 = 0                                                  # previous f0 to create f0 trajectory

	    fig = plt.figure(figsize = (10.5, 7.1), dpi = 100) 
	    ax0 = plt.subplot2grid((8,6), (0, 0), colspan = 6)
	    ax0.set_position([0.04, 0.955, 0.92, 0.015])
	    ax0.set_title("timeline", size = 7, fontweight = 'bold')
	    ax0.yaxis.set_ticks([])                           					 # no y axis ticks
	    ax0.xaxis.set_ticks([0, np.float32(x.size)/fs])              # set only two ticks in the limits of the plot
	    ax0.set_xticklabels(['0 s',  '%.2f' % (np.float32(x.size)/fs) + ' s'])
	    ax0.set_xlim(0, np.float32(x.size)/fs)
	    ax0.plot(time, np.zeros(x.size), lw = 1.5)
	    plt.tick_params(axis = 'both', labelsize = 8)
	    rect_zoom = patches.Rectangle((0, -2**7), width = (80.0*H)/fs, height = 2**15, color = 'black', alpha = 0.2)
	    ax0.add_patch(rect_zoom)

	    ax1 = plt.subplot2grid((8, 6), (1, 0), colspan = 6)
	    ax1.set_position([0.04, 0.87, 0.92, 0.05])
	    ax1.set_title("Input Signal (x)", size = 9, fontweight = 'bold')
	    ax1.locator_params(axis = 'y', nbins = 5)
	    ax1.set_xlim(0, (80.0*H)/fs)
	    ax1.set_ylim(x.min(), x.max())
	    plt.tick_params(axis = 'both', labelsize = 8)
	    plt.setp(ax1.get_xticklabels(), visible = False)
	    ax1.plot(time[:80*H], x[:80*H], 'b')

	    ax2 = plt.subplot2grid((8, 6), (2, 0), colspan = 6, sharex = ax1, sharey = ax1)
	    ax2.set_position([0.04, 0.79, 0.92, 0.05])
	    ax2.set_title("Output Signal (yh)", size = 9, fontweight = 'bold')
	    ax2.set_xlim(0, (80.0*H)/fs)
	    ax2.set_ylim(x.min(), x.max())
	    plt.tick_params(axis = 'both', labelsize = 8)

	    ax3 = plt.subplot2grid((8, 6), (3, 0), rowspan = 2, colspan = 3)
	    ax3.set_position([0.06, 0.52, 0.42, 0.21])
	    ax3.set_title("Original spectrum (mX, iploc, ipmag, f0, hloc, hmag)", size = 9, fontweight = 'bold')
	    ax3.set_xlabel("Frequency (Hz)", size = 8)
	    ax3.set_ylabel("Amplitude (dB)", size = 8)
	    ax3.set_xlim(0, freq_range)
	    ax3.set_ylim(-100, 0)
	    plt.tick_params(axis = 'both', labelsize = 8)

	    ax4 = plt.subplot2grid((8, 6), (3, 4), rowspan = 2, colspan = 3, sharex = ax3, sharey = ax3)
	    ax4.set_position([0.55, 0.52, 0.42, 0.21])
	    ax4.set_title("Harmonic plus residual spectrum (mXh, mXr, mX2)", size = 9, fontweight = 'bold')
	    ax4.set_xlabel("Frequency (Hz)", size = 8)
	    ax4.set_ylabel("Amplitude (dB)", size = 8)
	    ax4.set_xlim(0, freq_range)
	    ax4.set_ylim(-100, 0)
	    plt.tick_params(axis = 'both', labelsize = 8)

	    ax5 = plt.subplot2grid((8, 6), (7, 1), rowspan = 2, colspan = 4)
	    ax5.set_position([0.05, 0.03, 0.92, 0.42])
	    ax5.set_title("Peak tracking", size = 9, fontweight = 'bold')
	    ax5.imshow(specgram, interpolation = 'nearest', extent = (0, pend/H, 0, freq_range), aspect = 'auto', cmap = 'jet', vmin = -100, vmax = -20)
	    ax5.set_ylabel("Frequency (Hz)", size = 8)
	    ax5.set_xlim(0, 80)
	    ax5.set_ylim(0, freq_range)
	    ax5.ticklabel_format(axis = 'y', scilimits = (-2, 2))    # use scientific limits above 1e2
	    plt.tick_params(axis = 'both', labelsize = 8)

    while pin<pend:

    #-----analysis-----             
      
      xw = x[pin-hM:pin+hM-1] * w                                  # window the input sound

      fftbuffer = np.zeros(N)                                      # reset buffer
      fftbuffer[:hM] = xw[hM-1:]                                   # zero-phase window in fftbuffer
      fftbuffer[N-hM+1:] = xw[:hM-1]                           

      X = fft(fftbuffer)                                           # compute FFT
      mX = 20 * np.log10( abs(X[:hN]) )                            # magnitude spectrum of positive frequencies
      ploc = PP.peakDetection(mX, hN, t)                
      pX = np.unwrap( np.angle(X[:hN]) )                           # unwrapped phase spect. of positive freq.     
      iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)       # refine peak values

      if plot: specgram[:, n_frame] = mX[n_bins-1::-1]
      f0 = fd.f0DetectionTwm(iploc, ipmag, N, fs, f0et, minf0, maxf0)  # find f0

      hloc = np.zeros(nH)                                          # initialize harmonic locations
      hmag = np.zeros(nH)-100                                      # initialize harmonic magnitudes
      hphase = np.zeros(nH)                                        # initialize harmonic phases
      hf = (f0>0) * (f0*np.arange(1, nH+1))                        # initialize harmonic frequencies
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
      
      harmonics = np.float32(hloc)/N*fs
      hloc = (hloc!=0) * (hloc*Ns/N)                               # synth. locs
      
      ri = pin-hNs-1                                               # input sound pointer for residual analysis
      xw2 = x[ri:ri+Ns]*wr                                         # window the input sound                                       
      fftbuffer = np.zeros(Ns)                                     # reset buffer
      fftbuffer[:hNs] = xw2[hNs:]                                  # zero-phase window in fftbuffer
      fftbuffer[hNs:] = xw2[:hNs]                            
      X2 = fft(fftbuffer)                                          # compute FFT for residual analysis
      
      Xh = GS.genSpecSines(hloc, hmag, hphase, Ns)                 # generate sines
      Xr = X2-Xh                                                   # get the residual complex spectrum
      mXr = 20 * np.log10( abs(Xr[:hNs]) )                         # magnitude spectrum of residual
      mXrenv = resample(np.maximum(-200, mXr), mXr.size*stocf)     # decimate the magnitude spectrum and avoid -Inf

    #-----synthesis data-----
      
      yhloc = hloc                                                 # synthesis harmonics locs
      yhmag = hmag                                                 # synthesis harmonic amplitudes
      mYrenv = mXrenv                                              # synthesis residual envelope
      yf0 = f0  

    #-----transformations-----

    #-----synthesis-----
      
      yhphase += 2*np.pi * (lastyhloc+yhloc)/2/Ns*H                # propagate phases
      lastyhloc = yhloc 
      
      Yh = GS.genSpecSines(yhloc, yhmag, yhphase, Ns)              # generate spec sines 
      mYs = resample(mYrenv, hNs)                                  # interpolate to original size
      mYs = 10**(mYs/20)                                           # dB to linear magnitude  
      if f0>0:
          mYs *= np.cos(np.pi*np.arange(0, hNs)/Ns*fs/yf0)**2      # filter residual

      fc = 1+round(500.0/fs*Ns)                                    # 500 Hz
      mYs[:fc] *= (np.arange(0, fc)/(fc-1))**2                     # HPF
      pYs = 2*np.pi * np.random.rand(hNs)                          # generate phase random values
      
      Ys = np.zeros(Ns, dtype = complex)
      Ys[:hNs] = mYs * np.exp(1j*pYs)                              # generate positive freq.
      Ys[hNs+1:] = mYs[:0:-1] * np.exp(-1j*pYs[:0:-1])             # generate negative freq.

      fftbuffer = np.zeros(Ns)
      fftbuffer = np.real( ifft(Yh) )                            

      yhw[:hNs-1] = fftbuffer[hNs+1:]                              # sines in time domain using IFFT
      yhw[hNs-1:] = fftbuffer[:hNs+1] 

      fftbuffer = np.zeros(Ns)
      fftbuffer = np.real( ifft(Ys) )
      ysw[:hNs-1] = fftbuffer[hNs+1:]                              # stochastic in time domain using IFFT
      ysw[hNs-1:] = fftbuffer[:hNs+1]

      yh[ri:ri+Ns] += sw*yhw                                       # overlap-add for sines
      ys[ri:ri+Ns] += sws*ysw                                      # overlap-add for stoch

      #-----plotting-------
      if plot and n_frame>=nFrameStart and (n_frame%step == 0 or (pin+H)>pend):
        
	    # clear all plots

	      # clear only if not enough space to plot
	      if pin > ax1.get_xlim()[1]*fs - (5.0*H) :
	        clip_in = np.float32(pin) - 50.0*H
	        clip_spec = pin/H - 50.0
	        rect_zoom.remove()
	        rect_zoom = patches.Rectangle((clip_in/fs, -2**7), width = (80.0*H)/fs, height = 2**15, color = 'black', alpha = 0.2)
	        ax0.add_patch(rect_zoom)
	        
	        ax1.cla()
	        ax1.set_xlim(clip_in/fs, ((80.0*H)+clip_in)/fs)
	        ax1.set_ylim(x.min(), x.max())
	        ax1.set_title("Input Signal (x)", size = 9, fontweight = 'bold')
	        ax1.locator_params(axis = 'y', nbins = 5)
	        plt.setp(ax1.get_xticklabels(), visible = False)
	        ax1.plot(time[:clip_in+80*H], x[:clip_in+80*H], 'b')
	        
	        ax2.cla()
	        ax2.set_xlim(clip_in/fs, ((80.0*H)+clip_in)/fs)
	        ax2.set_ylim(x.min(), x.max())
	        ax2.set_title("Output Signal (yh)", size = 9, fontweight = 'bold')
	        ax2.locator_params(axis = 'y', nbins = 5)
	        ax2.plot(time[:ri], yh[:ri], 'b')
	        
	        ax5.set_xlim(clip_spec, clip_spec+80)

	      ax3.cla()
	      ax3.set_title("Original spectrum (mX, iploc, ipmag, f0, hloc, hmag)", size = 9, fontweight = 'bold')
	      ax3.set_xlabel("Frequency (Hz)", size = 8)
	      ax3.set_ylabel("Amplitude (dB)", size = 8)
	      ax3.set_xlim(0, freq_range)
	      ax3.set_ylim(-100, 0)

	      ax4.cla()
	      ax4.set_title("Harmonic plus residual spectrum (mXh, mXr, mX2)", size = 9, fontweight = 'bold')
	      ax4.set_xlabel("Frequency (Hz)", size = 8)
	      ax4.set_ylabel("Amplitude (dB)", size = 8)
	      ax4.set_xlim(0, freq_range)
	      ax4.set_ylim(-100, 0)

	    # plot all the information of the current sample

	      rect = patches.Rectangle((np.float32(pin-hM)/fs, -2**7), width = np.float32(w.size)/fs, height = 2**15, color = 'blue', alpha = 0.5)
	      ax1.add_patch(rect) 
	      if process: plt.draw()

	      ax3.plot(freq, mX[:n_bins], 'b')                                              # plot spectrum
	      ax3.fill_between(freq, -200, mX[:n_bins], facecolor = 'blue', alpha = 0.3) 
	      if process: plt.draw()
	      ax3.plot(np.float32(iploc[:n_bins])/N*fs, ipmag[:n_bins], 'rx', ms = 3)       # plot interpolated peak locations
	      if process: plt.draw()

	      ax5.imshow(specgram, interpolation = 'nearest', extent = (0, pend/H, 0, freq_range), aspect = 'auto', cmap = 'jet', vmin = -100, vmax = -20)
	      if process: plt.draw()

	      if f0 > 0:                                                  									# plot f0
	        loc = np.where(iploc/N*fs == f0)[0] 
	        if loc.size == 0: loc = np.argmin(np.abs(iploc/N*fs-f0))  									# closest peak location
	        ax3.plot(f0, ipmag[loc], 'go', ms = 4)                    									# plot in spectrum
	        if prev_f0 != 0 and f0 != 0:                              									# plot in spectrogram
	          ax5.plot([n_frame-0.5, n_frame+0.5], [prev_f0, f0], '-or', ms = 3, mfc = 'green', lw = 1.6)
	        elif prev_f0 == 0 and f0 != 0:                            									# initialize new line of f0's
	          ax5.plot(n_frame+0.5, f0, 'or', ms = 3, mfc = 'green')
	        if process: plt.draw()

	      if step == 1: prev_f0 = f0 												# save prev. f0 only if we are not rewinding plots

	      if f0 > 0: ax3.plot(harmonics[1:], hmag[1:], 'o', ms = 3, mfc = 'yellow') 		# plot harmonics
	      for i in range(1, nH-1):
	        if prev_harmonics[i] != 0 and harmonics[i] != 0: 
	          ax5.plot([n_frame-0.5, n_frame+0.5], [prev_harmonics[i], harmonics[i]], '-og', ms = 2.5, mfc = 'yellow', lw = 1.3)
	        elif prev_harmonics[i] == 0 and harmonics[i] != 0:        									# initialize new line of harmonics
	          ax5.plot(n_frame+0.5, harmonics[i], 'og', ms = 2.5, mfc = 'yellow')
	      
	      if process: plt.draw()

	      if step == 1: prev_harmonics = harmonics          # save prev. harmonics only if we are not rewinding plots

	      mX2 = 20 * np.log10( abs(X2[:hNs]) )                         # magnitude spectrum of positive frequencies
	      mX2 = resample(np.maximum(-200, mX2), hN)
	      ax4.plot(freq[:n_bins], mX2[:n_bins], 'b', alpha = 0.3)
	      ax4.fill_between(freq[:n_bins], -200, mX2[:n_bins], facecolor = 'blue', alpha = 0.1)
	      if process: plt.draw()

	      mXh = 20 * np.log10( abs(Xh[:hNs]) )                         # magnitude spectrum of positive frequencies
	      mXh = resample(np.maximum(-200, mXh), hN)
	      ax4.plot(freq[:n_bins], mXh[:n_bins], 'g')
	      ax4.fill_between(freq[:n_bins], -200, mXh[:n_bins], facecolor = 'green', alpha = 0.4)
	      if process: plt.draw()

	      mXr = resample(np.maximum(-200, mXr), hN)
	      ax4.plot(freq[:n_bins], mXr[:n_bins], 'r', alpha = 0.3)
	      ax4.fill_between(freq[:n_bins], -200, mXr[:n_bins], facecolor = 'red', alpha = 0.1)
	      if process: plt.draw()

	      rect2 = patches.Rectangle((np.float32(ri)/fs, -2**7), width = np.float32(Ns)/fs, height = 2**15, color = 'green', alpha = 0.3)
	      ax2.cla()
	      ax2.set_xlim(clip_in/fs, ((80.0*H)+clip_in)/fs)
	      ax2.set_ylim(x.min(), x.max())
	      ax2.set_title("Output Signal (yh)", size = 9, fontweight = 'bold')
	      ax2.locator_params(axis = 'y', nbins = 5)
	      ax2.add_patch(rect2)  
	      ax2.plot(time[:ri+Ns], yh[:ri+Ns], 'b')
	      plt.draw()
	      rect2.remove()
	      rect.remove()
        
      n_frame += 1              																	 # increment number of frames analyzed
      pin += H                                                     # advance sound pointer

    y = yh+ys
    return y, yh, ys


if __name__ == '__main__':
  
  app = QApplication.instance()
  if app == None: app = QApplication(sys.argv)
  
  (fs, x) = wp.wavread('../../sounds/oboe.wav')
  
  form = MainWindow(x, fs)
  form.show()
  app.exec_()