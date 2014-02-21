import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft
import time
import math
import sys, os, functools

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import waveIO as WIO
import peakProcessing as PP
import errorHandler as EH
import stftAnal, dftAnal

try:
  import genSpecSines_C as GS
  import twm_C as TWM
except ImportError:
  import genSpecSines as GS
  import twm as TWM
  EH.printWarning(1)

class Node():

    def __init__(self, frequency, magnitude=None, harmonic_number=None):
        self.frequency = frequency
        self.magnitude = magnitude
        self.harmonic_number = harmonic_number
        self.connections = []

    def connect(self, connection):
        self.connections.append(connection)

    def disconnect_all(self):
        for connection in self.connections:
            connection.disconnect()


class Connection():

    def __init__(self, harmonic_node, peak_node, distance):
        self.harmonic_node = harmonic_node
        self.peak_node = peak_node
        self.magnitude = self.peak_node.magnitude
        self.distance = distance
        self.connected = False
        self.connect()

    def connect(self):
        self.peak_node.connect(self)
        self.harmonic_node.connect(self)
        self.connected = True

    def disconnect(self):
        self.connected = False

    def cascade_disconnect(self):
        self.peak_node.disconnect_all()
        self.harmonic_node.disconnect_all()

def harmonic_peaks(frequencies, magnitudes, f0_estimation,
                   nharmonics=20, tolerance_percentage=20, fs=44100):
    '''
    This algorithm finds the harmonic peaks of a signal given its spectral peaks and its fundamental frequency.
    Pre: peaks are ordered in frequency ascending order and are not repeated
    Pre: peak frequencies are below nyquist frequency (fs/2)
    Pre: frequencies length = magnitudes length
    @param frequencies: the frequencies of the spectral peaks [Hz] (ascending order)
    @param magnitudes: the magnitudes of the spectral peaks (corresponding to the frequencies)
    @param f0_estimation: an estimate of the fundamental frequency of the signal [Hz]
    @param nharmonics: number of harmonics (multiples to generate) from the detected f0
    @param tolerance_percentage: a percentage of the deviation allowed to detect harmonics
    @param fs: sampling frequency. It is used to limit harmonics to nyquist frequency
    @return harmonic_frequencies: the frequencies of harmonic peaks [Hz]
    @return harmonic_magnitudes: the magnitudes of harmonic peaks
    @return harmonic_numbers: the harmonic number this array position corresponds to (f0 -> 1, f1 -> 2, etc.)
    '''

    if f0_estimation == 0 or frequencies.size == 0:
        return np.array([]), np.array([]), np.array([])

    # find closest peak to f0 | Cost: O(n) (could be reduced to log(n) using dicotomic-search)
    f0 = frequencies[np.argmin(abs(frequencies - f0_estimation))]
    tolerance = f0 * tolerance_percentage / 100.
    if abs(f0 - f0_estimation) > tolerance:
        print("Harmonic Peaks found the F0 peak with a deviation greater than tolerance")

    # We want to reduce nharmonics as much as posible since the cost of the algorithm depends directly on it
    # Limit harmonics to the nyquist frequency
    nharmonics = min(nharmonics, int(fs / 2. / f0))
    # Limit harmonics to the greatest possible harmonic close to the greatest detected peak
    nharmonics = min(nharmonics, int((frequencies[-1]+tolerance) / f0))

    if nharmonics < 2:  # there are no harmonics to detect
        return np.array([]), np.array([]), np.array([])

    # Generate ideal harmonics: multiples of f0
    ideal_harmonics = np.arange(f0, f0*nharmonics + 1, f0)  # Cost: O(k), k = nharmonics

    # Create detected peak nodes | Cost: O(n), n = len(frequencies)
    peak_nodes = []
    for freq, mag in zip(frequencies, magnitudes):
        peak_nodes.append(Node(freq, magnitude=mag))

    # Create ideal harmonics nodes and its connections to peak nodes | Cost: O(n*k)
    connections = []
    for harmonic_number, freq in enumerate(ideal_harmonics):
        harmonic_node = Node(freq, harmonic_number=harmonic_number)
        for peak_node in peak_nodes:
            distance = abs(harmonic_node.frequency - peak_node.frequency)
            if distance <= tolerance:
                connections.append(Connection(harmonic_node, peak_node, distance))

    # Sort connections using distance | Cost: O(n*log(n))
    connections = sorted(connections, key=attrgetter('distance', 'magnitude'))

    # Select peak nodes closer to harmonics | Cost: O(k^2 + 2kn - k)
    selected_peak_nodes = []
    for connection in connections:
        if connection.connected:
            connection.peak_node.harmonic_number = connection.harmonic_node.harmonic_number
            selected_peak_nodes.append(connection.peak_node)
            connection.cascade_disconnect()

    # Sort to preserve frequency ascending order (to comply with essentia post-conditions) | Cost: O(n*log(n))
    selected_peak_nodes = sorted(selected_peak_nodes, key=attrgetter('frequency'))

    # Create output arrays | Cost: O(n)
    harmonic_frequencies = essentia.zeros(len(selected_peak_nodes))
    harmonic_magnitudes = essentia.zeros(len(selected_peak_nodes))
    harmonic_numbers = essentia.zeros(len(selected_peak_nodes))
    for i, peak in enumerate(selected_peak_nodes):
        harmonic_frequencies[i] = peak.frequency
        harmonic_magnitudes[i] = peak.magnitude
        harmonic_numbers[i] = peak.harmonic_number

    return harmonic_frequencies, harmonic_magnitudes, harmonic_numbers

def harmonicDetection (ploc, pmag, pphase, f0, nH, maxhd, plocp, pmagp):
  # detection of the harmonics from a set of spectral peaks
  # ploc: peak locations, pmag: peak magnitudes, pphase: peak phases
  # f0: fundamental frequency, nH: number of harmonics,
  # maxhd: max. relative deviation in harmonic detection (ex: .2)
  # plocp: peak locations of previous frame, pmagp: peak magnitude of previous frame,
  # returns hloc: harmonic locations, hmag: harmonic magnitudes, hphase: harmonic phases
  hloc = np.zeros(nH)                                         # initialize harmonic locations
  hmag = np.zeros(nH)-100                                     # initialize harmonic magnitudes
  hphase = np.zeros(nH)                                       # initialize harmonic phases
  hf = (f0>0)*(f0*np.arange(1, nH+1))                         # initialize harmonic frequencies
  hi = 0                                                      # initialize harmonic index
  npeaks = ploc.size                                          # number of peaks found
  while f0>0 and hi<nH and hf[hi]<fs/2:                       # find harmonic peaks
    dev = min(abs(ploc/N*fs - hf[hi]))
    pei = np.argmin(abs(ploc/N*fs - hf[hi]))                  # closest peak
    if (hi==0 or not any(hloc[:hi]==ploc[pei])) and dev<maxhd*hf[hi] :
      hloc[hi] = ploc[pei]                                    # harmonic locations
      hmag[hi] = pmag[pei]                                    # harmonic magnitudes
      hphase[hi] = pphase[pei]                                # harmonic phases
    hi += 1                                                   # increase harmonic index
  return hloc, hmag, hphase

def harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, maxhd, maxnpeaksTwm=10):
  # Analysis of a sound using the sinusoidal harmonic model
  # x: input sound, fs: sampling rate, w: analysis window, 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # maxhd: max. relative deviation in harmonic detection (ex: .2)
  # maxnpeaksTwm: maximum number of peaks used for F0 detection
  # returns hloc, hmag, hphase
  hN = N/2                                                # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
  pin = hM1                                               # init sound pointer in middle of anal window          
  pend = x.size - hM1                                     # last sample to start a frame
  fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
  yh = np.zeros(Ns)                                       # initialize output sound frame
  y = np.zeros(x.size)                                    # initialize output array
  w = w / sum(w)                                          # normalize analysis window
  hlocp = hmagp = []
  while pin<pend:             
  #-----analysis----- 
    x1 = x[pin-hM1:pin+hM2]                               # select frame
    mX, pX = dftAnal.dftAnal(x1, w, N)                    # compute dft            
    ploc = PP.peakDetection(mX, hN, t)                    # detect peak locations   
    iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)   # refine peak values
    f0 = TWM.f0DetectionTwm(iploc, ipmag, N, fs, f0et, minf0, maxf0, maxnpeaksTwm)  # find f0
 #   hloc, hmag, hphase = harmonicDetection (iploc, ipmag, ipphase, f0, nH, maxhd, hlocp, hmagp) # find harmonics
    hloc, hmag, hphase = harmonic_peaks(iploc, ipmag, f0)
    hlocp = hloc
    hmagp = hmagp
    if pin == hM1: 
      xhloc = np.array([hloc])
      xhmag = np.array([hmag])
      xhphase = np.array([hphase])
    else:
      xhloc = np.vstack((xhloc,np.array([hloc])))
      xhmag = np.vstack((xhmag, np.array([hmag])))
      xhphase = np.vstack((xhphase, np.array([hphase])))
    pin += H                                              # advance sound pointer
  return xhloc, xhmag, xhphase

if __name__ == '__main__':
  (fs, x) = WIO.wavread('../../sounds/sax-phrase-short.wav')
  x = x[1*fs:1.5*fs]
  w = np.blackman(901)
  N = 1024
  t = -90
  nH = 40
  minf0 = 350
  maxf0 = 700
  f0et = 10
  maxhd = 0.2
  maxnpeaksTwm = 5
  Ns = 512
  H = Ns / 4

  mX, pX = stftAnal.stftAnal(x, fs, w, N, H)
  hloc, hmag, hphase = harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, maxhd, maxnpeaksTwm)
  maxplotbin = int(N*10000.0/fs)
  numFrames = int(mX[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs)                             
  binFreq = np.arange(maxplotbin+1)*float(fs)/N                         
  plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:maxplotbin+1]))
  plt.autoscale(tight=True)
  
  harms = hloc*np.less(hloc,maxplotbin)*float(fs)/N
  harms[harms==0] = np.nan
  numFrames = int(hloc[:,0].size)
  plt.plot(frmTime, harms, 'x', color='k')
  plt.autoscale(tight=True)
  plt.title('harmonics on spectrogram')
  plt.show()

