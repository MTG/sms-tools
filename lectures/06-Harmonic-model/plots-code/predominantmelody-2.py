import sys, csv, os
from essentia import *
from essentia.standard import *
from pylab import *
from numpy import *
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import stft as STFT
import utilFunctions as UF

filename = '../../../sounds/carnatic.wav'
hopSize = 128
frameSize = 2048
sampleRate = 44100
guessUnvoiced = True

run_windowing = Windowing(type='hann', zeroPadding=3*frameSize) # Hann window with x4 zero padding
run_spectrum = Spectrum(size=frameSize * 4)
run_spectral_peaks = SpectralPeaks(minFrequency=50,
                                   maxFrequency=10000,
                                   maxPeaks=100,
                                   sampleRate=sampleRate,
                                   magnitudeThreshold=0,
                                   orderBy="magnitude")
run_pitch_salience_function = PitchSalienceFunction(magnitudeThreshold=30)
run_pitch_salience_function_peaks = PitchSalienceFunctionPeaks(minFrequency=100, maxFrequency=300)
run_pitch_contours = PitchContours(hopSize=hopSize, peakFrameThreshold=0.8)
run_pitch_contours_melody = PitchContoursMelody(guessUnvoiced=guessUnvoiced,
                                                hopSize=hopSize)

pool = Pool();

audio = MonoLoader(filename = filename)()
audio = EqualLoudness()(audio)

for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
    frame = run_windowing(frame)
    spectrum = run_spectrum(frame)
    peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)
    
    salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
    salience_peaks_bins, salience_peaks_saliences = run_pitch_salience_function_peaks(salience)
    
    pool.add('allframes_salience_peaks_bins', salience_peaks_bins)
    pool.add('allframes_salience_peaks_saliences', salience_peaks_saliences)

contours_bins, contours_saliences, contours_start_times, duration = run_pitch_contours(
        pool['allframes_salience_peaks_bins'],
        pool['allframes_salience_peaks_saliences'])
pitch, confidence = run_pitch_contours_melody(contours_bins,
                                              contours_saliences,
                                              contours_start_times,
                                              duration)

yf0 = UF.sinewaveSynth(pitch, .6, hopSize, sampleRate)

figure(1, figsize=(9, 6))

mX, pX = STFT.stftAnal(audio, hamming(frameSize), frameSize, hopSize)
maxplotfreq = 3000.0
numFrames = int(mX[:,0].size)
frmTime = hopSize*arange(numFrames)/float(sampleRate)                             
binFreq = sampleRate*arange(frameSize*maxplotfreq/sampleRate)/frameSize                       
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:int(frameSize*maxplotfreq/sampleRate+1)]))
plt.autoscale(tight=True)

offset = .5 * frameSize/sampleRate
time = hopSize*arange(size(pitch))/float(sampleRate)
pitch[pitch==0]=nan
plot(time, pitch, color='k', linewidth = 2)

plt.title('mX + prominent melody (carnatic.wav)')
tight_layout()
savefig('predominantmelody-2.png')
UF.wavwrite(yf0, sampleRate, 'predominantmelody-2.wav')

show()
