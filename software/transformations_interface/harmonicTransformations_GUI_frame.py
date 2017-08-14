# GUI frame for the harmonicTransformations_function.py

try:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter 
    import tkFileDialog, tkMessageBox
except ImportError:
    # for Python3
    from tkinter import *  ## notice lowercase 't' in tkinter here
    from tkinter import filedialog as tkFileDialog
    from tkinter import messagebox as tkMessageBox
import sys, os
from scipy.io.wavfile import read
import numpy as np
import harmonicTransformations_function as hT
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
 
class HarmonicTransformations_frame:
  
	def __init__(self, parent):  
		 
		self.parent = parent        
		self.initUI()

	def initUI(self):

		choose_label = "inputFile:"
		Label(self.parent, text=choose_label).grid(row=0, column=0, sticky=W, padx=5, pady=(10,2))
 
		#TEXTBOX TO PRINT PATH OF THE SOUND FILE
		self.filelocation = Entry(self.parent)
		self.filelocation.focus_set()
		self.filelocation["width"] = 32
		self.filelocation.grid(row=0,column=0, sticky=W, padx=(70, 5), pady=(10,2))
		self.filelocation.delete(0, END)
		self.filelocation.insert(0, '../../sounds/vignesh.wav')

		#BUTTON TO BROWSE SOUND FILE
		open_file = Button(self.parent, text="...", command=self.browse_file) #see: def browse_file(self)
		open_file.grid(row=0, column=0, sticky=W, padx=(340, 6), pady=(10,2)) #put it beside the filelocation textbox
 
		#BUTTON TO PREVIEW SOUND FILE
		preview = Button(self.parent, text=">", command=lambda:UF.wavplay(self.filelocation.get()), bg="gray30", fg="white")
		preview.grid(row=0, column=0, sticky=W, padx=(385,6), pady=(10,2))

		## HARMONIC TRANSFORMATIONS ANALYSIS

		#ANALYSIS WINDOW TYPE
		wtype_label = "window:"
		Label(self.parent, text=wtype_label).grid(row=1, column=0, sticky=W, padx=5, pady=(10,2))
		self.w_type = StringVar()
		self.w_type.set("blackman") # initial value
		window_option = OptionMenu(self.parent, self.w_type, "rectangular", "hanning", "hamming", "blackman", "blackmanharris")
		window_option.grid(row=1, column=0, sticky=W, padx=(65,5), pady=(10,2))

		#WINDOW SIZE
		M_label = "M:"
		Label(self.parent, text=M_label).grid(row=1, column=0, sticky=W, padx=(180, 5), pady=(10,2))
		self.M = Entry(self.parent, justify=CENTER)
		self.M["width"] = 5
		self.M.grid(row=1,column=0, sticky=W, padx=(200,5), pady=(10,2))
		self.M.delete(0, END)
		self.M.insert(0, "1201")

		#FFT SIZE
		N_label = "N:"
		Label(self.parent, text=N_label).grid(row=1, column=0, sticky=W, padx=(255, 5), pady=(10,2))
		self.N = Entry(self.parent, justify=CENTER)
		self.N["width"] = 5
		self.N.grid(row=1,column=0, sticky=W, padx=(275,5), pady=(10,2))
		self.N.delete(0, END)
		self.N.insert(0, "2048")

		#THRESHOLD MAGNITUDE
		t_label = "t:"
		Label(self.parent, text=t_label).grid(row=1, column=0, sticky=W, padx=(330,5), pady=(10,2))
		self.t = Entry(self.parent, justify=CENTER)
		self.t["width"] = 5
		self.t.grid(row=1, column=0, sticky=W, padx=(348,5), pady=(10,2))
		self.t.delete(0, END)
		self.t.insert(0, "-90")

		#MIN DURATION SINUSOIDAL TRACKS
		minSineDur_label = "minSineDur:"
		Label(self.parent, text=minSineDur_label).grid(row=2, column=0, sticky=W, padx=(5, 5), pady=(10,2))
		self.minSineDur = Entry(self.parent, justify=CENTER)
		self.minSineDur["width"] = 5
		self.minSineDur.grid(row=2, column=0, sticky=W, padx=(87,5), pady=(10,2))
		self.minSineDur.delete(0, END)
		self.minSineDur.insert(0, "0.1")

		#MAX NUMBER OF HARMONICS
		nH_label = "nH:"
		Label(self.parent, text=nH_label).grid(row=2, column=0, sticky=W, padx=(145,5), pady=(10,2))
		self.nH = Entry(self.parent, justify=CENTER)
		self.nH["width"] = 5
		self.nH.grid(row=2, column=0, sticky=W, padx=(172,5), pady=(10,2))
		self.nH.delete(0, END)
		self.nH.insert(0, "100")

		#MIN FUNDAMENTAL FREQUENCY
		minf0_label = "minf0:"
		Label(self.parent, text=minf0_label).grid(row=2, column=0, sticky=W, padx=(227,5), pady=(10,2))
		self.minf0 = Entry(self.parent, justify=CENTER)
		self.minf0["width"] = 5
		self.minf0.grid(row=2, column=0, sticky=W, padx=(275,5), pady=(10,2))
		self.minf0.delete(0, END)
		self.minf0.insert(0, "130")

		#MAX FUNDAMENTAL FREQUENCY
		maxf0_label = "maxf0:"
		Label(self.parent, text=maxf0_label).grid(row=2, column=0, sticky=W, padx=(330,5), pady=(10,2))
		self.maxf0 = Entry(self.parent, justify=CENTER)
		self.maxf0["width"] = 5
		self.maxf0.grid(row=2, column=0, sticky=W, padx=(380,5), pady=(10,2))
		self.maxf0.delete(0, END)
		self.maxf0.insert(0, "300")

		#MAX ERROR ACCEPTED
		f0et_label = "f0et:"
		Label(self.parent, text=f0et_label).grid(row=3, column=0, sticky=W, padx=5, pady=(10,2))
		self.f0et = Entry(self.parent, justify=CENTER)
		self.f0et["width"] = 3
		self.f0et.grid(row=3, column=0, sticky=W, padx=(42,5), pady=(10,2))
		self.f0et.delete(0, END)
		self.f0et.insert(0, "7")

		#ALLOWED DEVIATION OF HARMONIC TRACKS
		harmDevSlope_label = "harmDevSlope:"
		Label(self.parent, text=harmDevSlope_label).grid(row=3, column=0, sticky=W, padx=(90,5), pady=(10,2))
		self.harmDevSlope = Entry(self.parent, justify=CENTER)
		self.harmDevSlope["width"] = 5
		self.harmDevSlope.grid(row=3, column=0, sticky=W, padx=(190,5), pady=(10,2))
		self.harmDevSlope.delete(0, END)
		self.harmDevSlope.insert(0, "0.01")

		#BUTTON TO DO THE ANALYSIS OF THE SOUND
		self.compute = Button(self.parent, text="Analysis/Synthesis", command=self.analysis, bg="dark red", fg="white")
		self.compute.grid(row=4, column=0, padx=5, pady=(10,5), sticky=W)
		
		#BUTTON TO PLAY ANALYSIS/SYNTHESIS OUTPUT
		self.output = Button(self.parent, text=">", command=lambda:UF.wavplay('output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_harmonicModel.wav'), bg="gray30", fg="white")
		self.output.grid(row=4, column=0, padx=(145,5), pady=(10,5), sticky=W)

		###
		#SEPARATION LINE
		Frame(self.parent,height=1,width=50,bg="black").grid(row=5, pady=5, sticky=W+E)
		###

		#FREQUENCY SCALING FACTORS
		freqScaling_label = "Frequency scaling factors (time, value pairs):"
		Label(self.parent, text=freqScaling_label).grid(row=6, column=0, sticky=W, padx=5, pady=(5,2))
		self.freqScaling = Entry(self.parent, justify=CENTER)
		self.freqScaling["width"] = 35
		self.freqScaling.grid(row=7, column=0, sticky=W+E, padx=5, pady=(0,2))
		self.freqScaling.delete(0, END)
		self.freqScaling.insert(0, "[0, 2.0, 1, 0.3]")

		#FREQUENCY STRETCHING FACTORSharmonicModelTransformation
		freqStretching_label = "Frequency stretching factors (time, value pairs):"
		Label(self.parent, text=freqStretching_label).grid(row=8, column=0, sticky=W, padx=5, pady=(5,2))
		self.freqStretching = Entry(self.parent, justify=CENTER)
		self.freqStretching["width"] = 35
		self.freqStretching.grid(row=9, column=0, sticky=W+E, padx=5, pady=(0,2))
		self.freqStretching.delete(0, END)
		self.freqStretching.insert(0, "[0, 1, 1, 1.5]")

		#TIMBRE PRESERVATION
		timbrePreservation_label = "Timbre preservation (1 preserves original timbre, 0 it does not):"
		Label(self.parent, text=timbrePreservation_label).grid(row=10, column=0, sticky=W, padx=5, pady=(5,2))
		self.timbrePreservation = Entry(self.parent, justify=CENTER)
		self.timbrePreservation["width"] = 2
		self.timbrePreservation.grid(row=10, column=0, sticky=W+E, padx=(395,5), pady=(5,2))
		self.timbrePreservation.delete(0, END)
		self.timbrePreservation.insert(0, "1")

		#TIME SCALING FACTORS
		timeScaling_label = "Time scaling factors (time, value pairs):"
		Label(self.parent, text=timeScaling_label).grid(row=11, column=0, sticky=W, padx=5, pady=(5,2))
		self.timeScaling = Entry(self.parent, justify=CENTER)
		self.timeScaling["width"] = 35
		self.timeScaling.grid(row=12, column=0, sticky=W+E, padx=5, pady=(0,2))
		self.timeScaling.delete(0, END)
		self.timeScaling.insert(0, "[0, 0, 0.671, 0.671, 1.978, 1.978+1.0]")

		#BUTTON TO DO THE SYNTHESIS
		self.compute = Button(self.parent, text="Apply Transformation", command=self.transformation_synthesis, bg="dark green", fg="white")
		self.compute.grid(row=13, column=0, padx=5, pady=(10,15), sticky=W)

		#BUTTON TO PLAY TRANSFORMATION SYNTHESIS OUTPUT
		self.transf_output = Button(self.parent, text=">", command=lambda:UF.wavplay('output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_harmonicModelTransformation.wav'), bg="gray30", fg="white")
		self.transf_output.grid(row=13, column=0, padx=(165,5), pady=(10,15), sticky=W)

		# define options for opening file
		self.file_opt = options = {}
		options['defaultextension'] = '.wav'
		options['filetypes'] = [('All files', '.*'), ('Wav files', '.wav')]
		options['initialdir'] = '../../sounds/'
		options['title'] = 'Open a mono audio file .wav with sample frequency 44100 Hz'

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
			nH = int(self.nH.get())
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

		except ValueError as errorMessage:
			tkMessageBox.showerror("Input values error", errorMessage)

		except AttributeError:
			tkMessageBox.showerror("Analysis not computed", "First you must analyse the sound!")
