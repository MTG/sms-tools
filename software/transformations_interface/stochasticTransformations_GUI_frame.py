# GUI frame for the stochasticTransformations_function.py

from Tkinter import *
import tkFileDialog, tkMessageBox
import sys, os
import pygame
from scipy.io.wavfile import read
import numpy as np
import stochasticTransformations_function as sT
 
class StochasticTransformations_frame:
  
	def __init__(self, parent):  
		 
		self.parent = parent        
		self.initUI()
		pygame.init()

	def initUI(self):

		choose_label = "inputFile:"
		Label(self.parent, text=choose_label).grid(row=0, column=0, sticky=W, padx=5, pady=(10,2))
 
		#TEXTBOX TO PRINT PATH OF THE SOUND FILE
		self.filelocation = Entry(self.parent)
		self.filelocation.focus_set()
		self.filelocation["width"] = 25
		self.filelocation.grid(row=0,column=0, sticky=W, padx=(70, 5), pady=(10,2))
		self.filelocation.delete(0, END)
		self.filelocation.insert(0, '../../sounds/rain.wav')

		#BUTTON TO BROWSE SOUND FILE
		self.open_file = Button(self.parent, text="...", command=self.browse_file) #see: def browse_file(self)
		self.open_file.grid(row=0, column=0, sticky=W, padx=(280, 6), pady=(10,2)) #put it beside the filelocation textbox
 
		#BUTTON TO PREVIEW SOUND FILE
		self.preview = Button(self.parent, text=">", command=self.preview_sound, bg="gray30", fg="white")
		self.preview.grid(row=0, column=0, sticky=W, padx=(325,6), pady=(10,2))

		## STOCHASTIC TRANSFORMATIONS ANALYSIS

		#DECIMATION FACTOR
		stocf_label = "stocf:"
		Label(self.parent, text=stocf_label).grid(row=1, column=0, sticky=W, padx=(5,5), pady=(10,2))
		self.stocf = Entry(self.parent, justify=CENTER)
		self.stocf["width"] = 5
		self.stocf.grid(row=1, column=0, sticky=W, padx=(47,5), pady=(10,2))
		self.stocf.delete(0, END)
		self.stocf.insert(0, "0.1")

		#TIME SCALING FACTORS
		timeScaling_label = "Time scaling factors, in time-value pairs:"
		Label(self.parent, text=timeScaling_label).grid(row=2, column=0, sticky=W, padx=5, pady=(5,2))
		self.timeScaling = Entry(self.parent, justify=CENTER)
		self.timeScaling.grid(row=3, column=0, sticky=W+E, padx=5, pady=(0,2))
		self.timeScaling.delete(0, END)
		self.timeScaling.insert(0, "[0, 0, 1, 2]")

		#BUTTON TO DO THE SYNTHESIS
		self.compute = Button(self.parent, text="Apply Transformation", command=self.transformation_synthesis, bg="dark green", fg="white")
		self.compute.grid(row=13, column=0, padx=5, pady=(10,15), sticky=W)

		#BUTTON TO PLAY TRANSFORMATION SYNTHESIS OUTPUT
		self.transf_output = Button(self.parent, text=">", command=lambda:self.play_out_sound('stochasticModelTransformation'), bg="gray30", fg="white")
		self.transf_output.grid(row=13, column=0, padx=(165,5), pady=(10,15), sticky=W)

		# define options for opening file
		self.file_opt = options = {}
		options['defaultextension'] = '.wav'
		options['filetypes'] = [('All files', '.*'), ('Wav files', '.wav')]
		options['initialdir'] = '../../sounds/'
		options['title'] = 'Open a mono audio file .wav with sample frequency 44100 Hz'

	def preview_sound(self):
		self.dummy = 2
		filename = self.filelocation.get()

		if filename[-4:] == '.wav':
			(fs, x) = read(filename)
		else:
			tkMessageBox.showerror("Wav file", "The audio file must be a .wav")
			return

		if len(x.shape) > 1 :
			tkMessageBox.showerror("Stereo file", "Audio file must be Mono not Stereo")
		elif fs != 44100:
			tkMessageBox.showerror("Sample Frequency", "Sample frequency must be 44100 Hz")
		else:
			sound = pygame.mixer.Sound(filename)
			sound.play()
 
	def browse_file(self):
		
		self.filename = tkFileDialog.askopenfilename(**self.file_opt)
 
		#set the text of the self.filelocation
		self.filelocation.delete(0, END)
		self.filelocation.insert(0,self.filename)

	def analysis(self):
		
		try:
			inputFile = self.filelocation.get()
			window =  self.w_type.get()
			M = int(self.M.get())
			N = int(self.N.get())
			t = int(self.t.get())
			minSineDur = float(self.minSineDur.get())
			maxnSines = int(self.maxnSines.get())
			minf0 = int(self.minf0.get())
			maxf0 = int(self.maxf0.get())
			f0et = int(self.f0et.get())
			harmDevSlope = float(self.harmDevSlope.get())

			self.inputFile, self.fs, self.hfreq, self.hmag = hT.analysis(inputFile, window, M, N, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope)

		except ValueError:
			tkMessageBox.showerror("Input values error", "Some parameters are incorrect")

	def transformation_synthesis(self):

		try:
			inputFile = self.inputFile
			fs =  self.fs
			hfreq = self.hfreq
			hmag = self.hmag
			freqScaling = np.array(eval(self.freqScaling.get()))
			freqStretching = np.array(eval(self.freqStretching.get()))
			timbrePreservation = int(self.timbrePreservation.get())
			timeScaling = np.array(eval(self.timeScaling.get()))

			hT.transformation_synthesis(inputFile, fs, hfreq, hmag, freqScaling, freqStretching, timbrePreservation, timeScaling)

		except ValueError:
			tkMessageBox.showerror("Input values error", "Some parameters are incorrect")

		except AttributeError:
			tkMessageBox.showerror("Analysis not computed", "First you must analyse the sound!")

	def play_out_sound(self, extension):

		filename = 'output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_' + extension + '.wav'
		if os.path.isfile(filename):
			sound = pygame.mixer.Sound(filename)
			sound.play()
		else:
			tkMessageBox.showerror("Output audio file not found", "The output audio file has not been computed yet")
