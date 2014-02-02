import wave
import pyaudio
import os
from scipy.io.wavfile import write

def play(x, fs):

	CHUNK = 1024
	write('temp_file.wav', fs, x)
	wf = wave.open('temp_file.wav', 'rb')
	p = pyaudio.PyAudio()

	stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(), rate=wf.getframerate(), output=True)

	data = wf.readframes(CHUNK)

	while data is not '':
		stream.write(data)
		data = wf.readframes(CHUNK)

	stream.stop_stream()
	stream.close()

	p.terminate()
	os.remove(os.getcwd()+'/temp_file.wav')