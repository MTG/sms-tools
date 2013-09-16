import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift


import sys, os, functools

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



def sps(x, fs, w, N, t, maxnS, stocf) :
  # Analysis/synthesis of a sound using the sinusoidal plus stochastic model
  # x: input sound, fs: sampling rate, w: analysis window (odd size), 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # maxnS: maximum number of sinusoids,
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
  yhw = np.zeros(Ns)                                            # initialize output sine sound frame
  ysw = np.zeros(Ns)                                            # initialize output residual sound frame
  yh = np.zeros(x.size)                                         # initialize output sine component
  ys = np.zeros(x.size)                                         # initialize output residual component
  w = w / sum(w)                                                # normalize analysis window
  sw = np.zeros(Ns)     
  ow = triang(2*H)                                              # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                       # synthesis window
  bh = bh / sum(bh)                                             # normalize synthesis window
  wr = bh                                                       # window for residual
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
  sws = H*hanning(Ns)/2                                         # synthesis window for stochastic
  lastysloc = np.zeros(maxnS)                                   # initialize synthesis harmonic locations
  ysphase = 2*np.pi * np.random.rand(maxnS)                     # initialize synthesis harmonic phases
  fridx = 0                                                     # frame pointer
  isInitFrame = True                                            # True for frames equivalent to initial frame (for synth part)
  lastnS = 0                                                    # it doesnot harm to initialize this variable with 0.

  #-----initialize plots----- 

  clip_in = 0.0                                                 # samples to clip input/output signal
  clip_spec = 0.0                                               # number of frames to clip spectrogram
  freq = np.arange(0, freq_range, fs/N)                         # frequency axis in Hz
  freq = freq[:freq.size-1]
  time = np.arange(0, np.float32(x.size)/fs, 1.0/fs)            # time axis in seconds
  n_frame = 0
  n_bins = freq.size
  specgram = np.ones((n_bins, pend/H)) * -200                   # initialize spectrogram
  prev_peaks_loc = np.zeros(maxnS)                              # harmonic trajectories


  fig = plt.figure(figsize = (10.5, 7.1), dpi = 100)
  ax0 = plt.subplot2grid((8,6), (0, 0), colspan = 6)
  ax0.set_position([0.04, 0.955, 0.92, 0.015])
  ax0.set_title("timeline", size = 7, fontweight = 'bold')
  ax0.yaxis.set_ticks([])                           # no y axis ticks
  ax0.xaxis.set_ticks([0, np.float32(x.size)/fs])
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
    
    if fridx==0 or lastnS==0 :     # whenever lastnS is zero implies frame is equivalent to initial frame
      isInitFrame = True

  #-----analysis-----             
    xw = x[pin-hM:pin+hM-1] * w                                  # window the input sound
    fftbuffer = np.zeros(N)                                      # reset buffer
    fftbuffer[:hM] = xw[hM-1:]                                   # zero-phase window in fftbuffer
    fftbuffer[N-hM+1:] = xw[:hM-1]              
    X = fft(fftbuffer)                                           # compute FFT
    mX = 20 * np.log10( abs(X[:hN]) )                            # magnitude spectrum of positive frequencies
    ploc = PP.peak_detection(mX, hN, t)                
    pX = np.unwrap( np.angle(X[:hN]) )                           # unwrapped phase spect. of positive freq.     
    iploc, ipmag, ipphase = PP.peak_interp(mX, pX, ploc)            # refine peak values
    
    smag = np.sort(ipmag)[::-1]                                  # sort peaks by magnitude in descending order
    I = np.argsort(ipmag)[::-1]
    
    nS = min(maxnS, np.where(smag>t)[0].size)                    # get peaks above threshold
    sloc = iploc[I[:nS]]
    sphase = ipphase[I[:nS]]  
    if isInitFrame :                                             # update last frame data
      lastnS = nS
      lastsloc = sloc
      lastsmag = smag
      lastsphase = sphase

    peaks_loc = np.float32(sloc)/N*fs
    sloc = (sloc!=0) * (sloc*Ns/N)                               # peak locations for synthesis
    lastidx = np.zeros(nS, dtype = int)
    for i in range(0, nS) :  # find closest peak to create trajectories
      idx = np.argmin(abs(sloc[i] - lastsloc[:lastnS]))  
      lastidx[i] = idx

    ri = pin-hNs-1                                               # input sound pointer for residual analysis
    xw2 = x[ri:ri+Ns]*wr                                         # window the input sound                                       
    fftbuffer = np.zeros(Ns)                                     # reset buffer
    fftbuffer[:hNs] = xw2[hNs:]                                  # zero-phase window in fftbuffer
    fftbuffer[hNs:] = xw2[:hNs]              
    X2 = fft(fftbuffer)                                          # compute FFT for residual analysis
    
    Xh = GS.genspecsines(sloc, smag, sphase, Ns)                    # generate sines
    Xr = X2-Xh                                                   # get the residual complex spectrum
    mXr = 20 * np.log10( abs(Xr[:hNs]) )                         # magnitude spectrum of residual
    mXrenv = resample(np.maximum(-200, mXr), mXr.size*stocf)     # decimate the magnitude spectrum and avoid -Inf                     

  #-----synthesis data-----
    ysloc = sloc                                                 # synthesis harmonics locs
    ysmag = smag[:nS]                                            # synthesis harmonic amplitudes
    mYrenv = mXrenv                                              # synthesis residual envelope

  #-----transformations-----

  #-----synthesis-----
    
    if isInitFrame :
      # Variables need to be initialized like for the first frame
      lastysloc = np.zeros(maxnS)                     # initialize synthesis harmonic locations
      ysphase = 2*np.pi * np.random.rand(maxnS)       # initialize synthesis harmonic phases
      
      lastysphase = ysphase                           # phase for first frame
    
    if nS>lastnS :                                    # initialize peaks that start
      lastysphase = np.concatenate((lastysphase, np.zeros(nS-lastnS)))
      lastysloc = np.concatenate((lastysloc, np.zeros(nS-lastnS)))
    
    ysphase = lastysphase[lastidx] + 2*np.pi*(lastysloc[lastidx]+ysloc)/2/Ns*H # propagate phases
    
    lastysloc = ysloc
    lastysphase = ysphase  
    lastnS = nS                                       # update last frame data
    lastsloc = sloc                                   # update last frame data
    lastsmag = smag                                   # update last frame data
    lastsphase = sphase                               # update last frame data

    Yh = GS.genspecsines(ysloc, ysmag, ysphase, Ns)      # generate spec sines 
    mYs = resample(mYrenv, hNs)                       # interpolate to original size
    pYs = 2*np.pi*np.random.rand(hNs)                 # generate phase random values
    
    Ys = np.zeros(Ns, dtype = complex)
    Ys[:hNs] = 10**(mYs/20) * np.exp(1j*pYs)                   # generate positive freq.
    Ys[hNs+1:] = 10**(mYs[:0:-1]/20) * np.exp(-1j*pYs[:0:-1])  # generate negative freq.

    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real( ifft(Yh) )                            
    yhw[:hNs-1] = fftbuffer[hNs+1:]                   # sines in time domain using IFFT
    yhw[hNs-1:] = fftbuffer[:hNs+1] 
    
    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real( ifft(Ys) )
    ysw[:hNs-1] = fftbuffer[hNs+1:]                   # stochastic in time domain using IFFT
    ysw[hNs-1:] = fftbuffer[:hNs+1]

    yh[ri:ri+Ns] += sw*yhw                            # overlap-add for sines
    ys[ri:ri+Ns] += sws*ysw                           # overlap-add for stoch
    
    #-----plotting-------
    # if n_frame > 1130 :
      
    # clear all plots
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

    rect = patches.Rectangle((np.float32(pin-hM)/fs, -2**7), width = np.float32(w.size)/fs, height = 2**15, color = 'blue', alpha = 0.5)
    ax1.add_patch(rect) 
    # plt.draw()

    # plot the sample

    ax3.plot(freq, mX[:n_bins], 'b')                                              # plot spectrum
    ax3.fill_between(freq, -200, mX[:n_bins], facecolor = 'blue', alpha = 0.3) 
    # plt.draw()
    ax3.plot(np.float32(iploc[:n_bins])/N*fs, ipmag[:n_bins], 'rx', ms = 3)       # plot interpolated peak locations
    # plt.draw()

    ax5.imshow(specgram, interpolation = 'nearest', extent = (0, pend/H, 0, freq_range), aspect = 'auto', cmap = 'jet', vmin = -100, vmax = -20)
    # plt.draw()

    ax3.plot(peaks_loc, smag[:nS], 'o', ms = 3, mfc = 'yellow') # plot harmonics
    for i in range(0, nS):
      ax5.plot([n_frame-0.5, n_frame+0.5], [prev_peaks_loc[lastidx[i]], peaks_loc[i]], '-og', ms = 2.5, mfc = 'yellow', lw = 1.3)
    
    prev_peaks_loc = peaks_loc
    # plt.draw()
    

    mX2 = 20 * np.log10( abs(X2[:hNs]) )                         # magnitude spectrum of positive frequencies
    mX2 = resample(np.maximum(-200, mX2), hN)
    ax4.plot(freq[:n_bins], mX2[:n_bins], 'b', alpha = 0.3)
    ax4.fill_between(freq[:n_bins], -200, mX2[:n_bins], facecolor = 'blue', alpha = 0.1)
    # plt.draw()

    mXh = 20 * np.log10( abs(Xh[:hNs]) )                         # magnitude spectrum of positive frequencies
    mXh = resample(np.maximum(-200, mXh), hN)
    ax4.plot(freq[:n_bins], mXh[:n_bins], 'g')
    ax4.fill_between(freq[:n_bins], -200, mXh[:n_bins], facecolor = 'green', alpha = 0.4)
    # plt.draw()

    mXr = resample(np.maximum(-200, mXr), hN)
    ax4.plot(freq[:n_bins], mXr[:n_bins], 'r', alpha = 0.3)
    ax4.fill_between(freq[:n_bins], -200, mXr[:n_bins], facecolor = 'red', alpha = 0.1)
    # plt.draw()

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
      
    n_frame += 1
    pin += H                                          # advance sound pointer
    fridx += 1                                        # advance frame pointer
    isInitFrame = False                               # variable meaningful only for current frame,
                                                      # therefore False at each frame
  y = yh+ys
  return y, yh, ys


if __name__ == '__main__':

    (fs, x) = wp.wavread('../../sounds/speech-female.wav')
    # wp.play(x, fs)

    # fig = plt.figure()
    w = np.hamming(801)
    N = 1024
    t = -120
    maxnS = 30
    stocf = 0.5
    y, yh, ys = sps(x, fs, w, N, t, maxnS, stocf)

    wp.play(y, fs)
    wp.play(yh, fs)
    wp.play(ys, fs)