# GUI frame for the hpsMorph_function.py

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
import hpsMorph_function as hM
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
 
class HpsMorph_frame:
  
	def __init__(self, parent):  
		 
		self.parent = parent        
		self.initUI()

	def initUI(self):

		## INPUT FILE 1
		choose1_label = "inputFile1:"
		Label(self.parent, text=choose1_label).grid(row=0, column=0, sticky=W, padx=5, pady=(10,2))
 
		#TEXTBOX TO PRINT PATH OF THE SOUND FILE
		self.filelocation1 = Entry(self.parent)
		self.filelocation1.focus_set()
		self.filelocation1["width"] = 30
		self.filelocation1.grid(row=0,column=0, sticky=W, padx=(75, 5), pady=(10,2))
		self.filelocation1.delete(0, END)
		self.filelocation1.insert(0, '../../sounds/violin-B3.wav')

		#BUTTON TO BROWSE SOUND FILE 1
		open_file1 = Button(self.parent, text="...", command=self.browse_file1) #see: def browse_file(self)
		open_file1.grid(row=0, column=0, sticky=W, padx=(330, 6), pady=(10,2)) #put it beside the filelocation textbox
 
		#BUTTON TO PREVIEW SOUND FILE 1
		preview1 = Button(self.parent, text=">", command=lambda:UF.wavplay(self.filelocation1.get()), bg="gray30", fg="white")
		preview1.grid(row=0, column=0, sticky=W, padx=(375,6), pady=(10,2))
		
		#ANALYSIS WINDOW TYPE SOUND 1
		wtype1_label = "window1:"
		Label(self.parent, text=wtype1_label).grid(row=1, column=0, sticky=W, padx=5, pady=(4,2))
		self.w1_type = StringVar()
		self.w1_type.set("blackman") # initial value
		window1_option = OptionMenu(self.parent, self.w1_type, "rectangular", "hanning", "hamming", "blackman", "blackmanharris")
		window1_option.grid(row=1, column=0, sticky=W, padx=(68,5), pady=(4,2))

		#WINDOW SIZE SOUND 1
		M1_label = "M1:"
		Label(self.parent, text=M1_label).grid(row=1, column=0, sticky=W, padx=(180, 5), pady=(4,2))
		self.M1 = Entry(self.parent, justify=CENTER)
		self.M1["width"] = 5
		self.M1.grid(row=1,column=0, sticky=W, padx=(208,5), pady=(4,2))
		self.M1.delete(0, END)
		self.M1.insert(0, "1001")

		#FFT SIZE SOUND 1
		N1_label = "N1:"
		Label(self.parent, text=N1_label).grid(row=1, column=0, sticky=W, padx=(265, 5), pady=(4,2))
		self.N1 = Entry(self.parent, justify=CENTER)
		self.N1["width"] = 5
		self.N1.grid(row=1,column=0, sticky=W, padx=(290,5), pady=(4,2))
		self.N1.delete(0, END)
		self.N1.insert(0, "1024")

		#THRESHOLD MAGNITUDE SOUND 1
		t1_label = "t1:"
		Label(self.parent, text=t1_label).grid(row=1, column=0, sticky=W, padx=(343,5), pady=(4,2))
		self.t1 = Entry(self.parent, justify=CENTER)
		self.t1["width"] = 5
		self.t1.grid(row=1, column=0, sticky=W, padx=(370,5), pady=(4,2))
		self.t1.delete(0, END)
		self.t1.insert(0, "-100")

		#MIN DURATION SINUSOIDAL TRACKS SOUND 1
		minSineDur1_label = "minSineDur1:"
		Label(self.parent, text=minSineDur1_label).grid(row=2, column=0, sticky=W, padx=(5, 5), pady=(4,2))
		self.minSineDur1 = Entry(self.parent, justify=CENTER)
		self.minSineDur1["width"] = 5
		self.minSineDur1.grid(row=2, column=0, sticky=W, padx=(92,5), pady=(4,2))
		self.minSineDur1.delete(0, END)
		self.minSineDur1.insert(0, "0.05")

		#MIN FUNDAMENTAL FREQUENCY SOUND 1
		minf01_label = "minf01:"
		Label(self.parent, text=minf01_label).grid(row=2, column=0, sticky=W, padx=(157,5), pady=(4,2))
		self.minf01 = Entry(self.parent, justify=CENTER)
		self.minf01["width"] = 5
		self.minf01.grid(row=2, column=0, sticky=W, padx=(208,5), pady=(4,2))
		self.minf01.delete(0, END)
		self.minf01.insert(0, "200")

		#MAX FUNDAMENTAL FREQUENCY SOUND 1
		maxf01_label = "maxf01:"
		Label(self.parent, text=maxf01_label).grid(row=2, column=0, sticky=W, padx=(270,5), pady=(4,2))
		self.maxf01 = Entry(self.parent, justify=CENTER)
		self.maxf01["width"] = 5
		self.maxf01.grid(row=2, column=0, sticky=W, padx=(325,5), pady=(4,2))
		self.maxf01.delete(0, END)
		self.maxf01.insert(0, "300")

		#MAX ERROR ACCEPTED SOUND 1
		f0et1_label = "f0et1:"
		Label(self.parent, text=f0et1_label).grid(row=3, column=0, sticky=W, padx=5, pady=(4,2))
		self.f0et1 = Entry(self.parent, justify=CENTER)
		self.f0et1["width"] = 3
		self.f0et1.grid(row=3, column=0, sticky=W, padx=(45,5), pady=(4,2))
		self.f0et1.delete(0, END)
		self.f0et1.insert(0, "10")

		#ALLOWED DEVIATION OF HARMONIC TRACKS SOUND 1
		harmDevSlope1_label = "harmDevSlope1:"
		Label(self.parent, text=harmDevSlope1_label).grid(row=3, column=0, sticky=W, padx=(108,5), pady=(4,2))
		self.harmDevSlope1 = Entry(self.parent, justify=CENTER)
		self.harmDevSlope1["width"] = 5
		self.harmDevSlope1.grid(row=3, column=0, sticky=W, padx=(215,5), pady=(4,2))
		self.harmDevSlope1.delete(0, END)
		self.harmDevSlope1.insert(0, "0.01")

		###
		#SEPARATION LINE
		Frame(self.parent,height=1,width=50,bg="black").grid(row=4, pady=5, sticky=W+E)
		###

		## INPUT FILE 2
		choose2_label = "inputFile2:"
		Label(self.parent, text=choose2_label).grid(row=5, column=0, sticky=W, padx=5, pady=(2,2))
 
		#TEXTBOX TO PRINT PATH OF THE SOUND FILE
		self.filelocation2 = Entry(self.parent)
		self.filelocation2.focus_set()
		self.filelocation2["width"] = 30
		self.filelocation2.grid(row=5,column=0, sticky=W, padx=(75, 5), pady=(2,2))
		self.filelocation2.delete(0, END)
		self.filelocation2.insert(0, '../../sounds/soprano-E4.wav')

		#BUTTON TO BROWSE SOUND FILE 2
		open_file2 = Button(self.parent, text="...", command=self.browse_file2) #see: def browse_file(self)
		open_file2.grid(row=5, column=0, sticky=W, padx=(330, 6), pady=(2,2)) #put it beside the filelocation textbox
 
		#BUTTON TO PREVIEW SOUND FILE 2
		preview2 = Button(self.parent, text=">", command=lambda:UF.wavplay(self.filelocation2.get()), bg="gray30", fg="white")
		preview2.grid(row=5, column=0, sticky=W, padx=(375,6), pady=(2,2))


		#ANALYSIS WINDOW TYPE SOUND 2
		wtype2_label = "window2:"
		Label(self.parent, text=wtype2_label).grid(row=6, column=0, sticky=W, padx=5, pady=(4,2))
		self.w2_type = StringVar()
		self.w2_type.set("hamming") # initial value
		window2_option = OptionMenu(self.parent, self.w2_type, "rectangular", "hanning", "hamming", "blackman", "blackmanharris")
		window2_option.grid(row=6, column=0, sticky=W, padx=(68,5), pady=(4,2))

		#WINDOW SIZE SOUND 2
		M2_label = "M2:"
		Label(self.parent, text=M2_label).grid(row=6, column=0, sticky=W, padx=(180, 5), pady=(4,2))
		self.M2 = Entry(self.parent, justify=CENTER)
		self.M2["width"] = 5
		self.M2.grid(row=6,column=0, sticky=W, padx=(208,5), pady=(4,2))
		self.M2.delete(0, END)
		self.M2.insert(0, "901")

		#FFT SIZE SOUND 2
		N2_label = "N2:"
		Label(self.parent, text=N2_label).grid(row=6, column=0, sticky=W, padx=(265, 5), pady=(4,2))
		self.N2 = Entry(self.parent, justify=CENTER)
		self.N2["width"] = 5
		self.N2.grid(row=6,column=0, sticky=W, padx=(290,5), pady=(4,2))
		self.N2.delete(0, END)
		self.N2.insert(0, "1024")

		#THRESHOLD MAGNITUDE SOUND 2
		t2_label = "t2:"
		Label(self.parent, text=t2_label).grid(row=6, column=0, sticky=W, padx=(343,5), pady=(4,2))
		self.t2 = Entry(self.parent, justify=CENTER)
		self.t2["width"] = 5
		self.t2.grid(row=6, column=0, sticky=W, padx=(370,5), pady=(4,2))
		self.t2.delete(0, END)
		self.t2.insert(0, "-100")

		#MIN DURATION SINUSOIDAL TRACKS SOUND 2
		minSineDur2_label = "minSineDur2:"
		Label(self.parent, text=minSineDur2_label).grid(row=7, column=0, sticky=W, padx=(5, 5), pady=(4,2))
		self.minSineDur2 = Entry(self.parent, justify=CENTER)
		self.minSineDur2["width"] = 5
		self.minSineDur2.grid(row=7, column=0, sticky=W, padx=(92,5), pady=(4,2))
		self.minSineDur2.delete(0, END)
		self.minSineDur2.insert(0, "0.05")

		#MIN FUNDAMENTAL FREQUENCY SOUND 2
		minf02_label = "minf02:"
		Label(self.parent, text=minf02_label).grid(row=7, column=0, sticky=W, padx=(157,5), pady=(4,2))
		self.minf02 = Entry(self.parent, justify=CENTER)
		self.minf02["width"] = 5
		self.minf02.grid(row=7, column=0, sticky=W, padx=(208,5), pady=(4,2))
		self.minf02.delete(0, END)
		self.minf02.insert(0, "250")

		#MAX FUNDAMENTAL FREQUENCY SOUND 2
		maxf02_label = "maxf02:"
		Label(self.parent, text=maxf02_label).grid(row=7, column=0, sticky=W, padx=(270,5), pady=(4,2))
		self.maxf02 = Entry(self.parent, justify=CENTER)
		self.maxf02["width"] = 5
		self.maxf02.grid(row=7, column=0, sticky=W, padx=(325,5), pady=(4,2))
		self.maxf02.delete(0, END)
		self.maxf02.insert(0, "500")

		#MAX ERROR ACCEPTED SOUND 2
		f0et2_label = "f0et2:"
		Label(self.parent, text=f0et2_label).grid(row=8, column=0, sticky=W, padx=5, pady=(4,2))
		self.f0et2 = Entry(self.parent, justify=CENTER)
		self.f0et2["width"] = 3
		self.f0et2.grid(row=8, column=0, sticky=W, padx=(45,5), pady=(4,2))
		self.f0et2.delete(0, END)
		self.f0et2.insert(0, "10")

		#ALLOWED DEVIATION OF HARMONIC TRACKS SOUND 2
		harmDevSlope2_label = "harmDevSlope2:"
		Label(self.parent, text=harmDevSlope2_label).grid(row=8, column=0, sticky=W, padx=(108,5), pady=(4,2))
		self.harmDevSlope2 = Entry(self.parent, justify=CENTER)
		self.harmDevSlope2["width"] = 5
		self.harmDevSlope2.grid(row=8, column=0, sticky=W, padx=(215,5), pady=(4,2))
		self.harmDevSlope2.delete(0, END)
		self.harmDevSlope2.insert(0, "0.01")

		###
		#SEPARATION LINE
		Frame(self.parent,height=1,width=50,bg="black").grid(row=9, pady=5, sticky=W+E)
		###

		#MAX NUMBER OF HARMONICS SOUND 1
		nH_label = "nH:"
		Label(self.parent, text=nH_label).grid(row=10, column=0, sticky=W, padx=(5,5), pady=(2,2))
		self.nH = Entry(self.parent, justify=CENTER)
		self.nH["width"] = 5
		self.nH.grid(row=10, column=0, sticky=W, padx=(35,5), pady=(2,2))
		self.nH.delete(0, END)
		self.nH.insert(0, "60")

		#DECIMATION FACTOR SOUND 1
		stocf_label = "stocf:"
		Label(self.parent, text=stocf_label).grid(row=10, column=0, sticky=W, padx=(98,5), pady=(2,2))
		self.stocf = Entry(self.parent, justify=CENTER)
		self.stocf["width"] = 5
		self.stocf.grid(row=10, column=0, sticky=W, padx=(138,5), pady=(2,2))
		self.stocf.delete(0, END)
		self.stocf.insert(0, "0.1")

		#BUTTON TO DO THE ANALYSIS OF THE SOUND
		self.compute = Button(self.parent, text="Analysis", command=self.analysis, bg="dark red", fg="white")
		self.compute.grid(row=10, column=0, padx=(210, 5), pady=(2,2), sticky=W)

		###
		#SEPARATION LINE
		Frame(self.parent,height=1,width=50,bg="black").grid(row=11, pady=5, sticky=W+E)
		###

		#
		hfreqIntp_label = "harmonic frequencies interpolation factors, 0 to 1 (time,value pairs)"
		Label(self.parent, text=hfreqIntp_label).grid(row=12, column=0, sticky=W, padx=5, pady=(2,2))
		self.hfreqIntp = Entry(self.parent, justify=CENTER)
		self.hfreqIntp["width"] = 35
		self.hfreqIntp.grid(row=13, column=0, sticky=W+E, padx=5, pady=(0,2))
		self.hfreqIntp.delete(0, END)
		self.hfreqIntp.insert(0, "[0, 0, .1, 0, .9, 1, 1, 1]")

		#
		hmagIntp_label = "harmonic magnitudes interpolation factors, 0 to 1 (time,value pairs)"
		Label(self.parent, text=hmagIntp_label).grid(row=14, column=0, sticky=W, padx=5, pady=(5,2))
		self.hmagIntp = Entry(self.parent, justify=CENTER)
		self.hmagIntp["width"] = 35
		self.hmagIntp.grid(row=15, column=0, sticky=W+E, padx=5, pady=(0,2))
		self.hmagIntp.delete(0, END)
		self.hmagIntp.insert(0, "[0, 0, .1, 0, .9, 1, 1, 1]")

		#
		stocIntp_label = "stochastic component interpolation factors, 0 to 1 (time,value pairs)"
		Label(self.parent, text=stocIntp_label).grid(row=16, column=0, sticky=W, padx=5, pady=(5,2))
		self.stocIntp = Entry(self.parent, justify=CENTER)
		self.stocIntp["width"] = 35
		self.stocIntp.grid(row=17, column=0, sticky=W+E, padx=5, pady=(0,2))
		self.stocIntp.delete(0, END)
		self.stocIntp.insert(0, "[0, 0, .1, 0, .9, 1, 1, 1]")


		#BUTTON TO DO THE SYNTHESIS
		self.compute = Button(self.parent, text="Apply Transformation", command=self.transformation_synthesis, bg="dark green", fg="white")
		self.compute.grid(row=18, column=0, padx=5, pady=(10,15), sticky=W)

		#BUTTON TO PLAY TRANSFORMATION SYNTHESIS OUTPUT
		self.transf_output = Button(self.parent, text=">", command=lambda:UF.wavplay('output_sounds/' + os.path.basename(self.filelocation1.get())[:-4] + '_hpsMorph.wav'), bg="gray30", fg="white")
		self.transf_output.grid(row=18, column=0, padx=(165,5), pady=(10,15), sticky=W)

		# define options for opening file
		self.file_opt = options = {}
		options['defaultextension'] = '.wav'
		options['filetypes'] = [('All files', '.*'), ('Wav files', '.wav')]
		options['initialdir'] = '../../sounds/'
		options['title'] = 'Open a mono audio file .wav with sample frequency 44100 Hz'
 
	def browse_file1(self):
		
		self.filename1 = tkFileDialog.askopenfilename(**self.file_opt)
 
		#set the text of the self.filelocation
		self.filelocation1.delete(0, END)
		self.filelocation1.insert(0,self.filename1)
 
	def browse_file2(self):
		
		self.filename2 = tkFileDialog.askopenfilename(**self.file_opt)
 
		#set the text of the self.filelocation
		self.filelocation2.delete(0, END)
		self.filelocation2.insert(0,self.filename2)

	def analysis(self):
		
		try:
			inputFile1 = self.filelocation1.get()
			window1 =  self.w1_type.get()
			M1 = int(self.M1.get())
			N1 = int(self.N1.get())
			t1 = int(self.t1.get())
			minSineDur1 = float(self.minSineDur1.get())
			minf01 = int(self.minf01.get())
			maxf01 = int(self.maxf01.get())
			f0et1 = int(self.f0et1.get())
			harmDevSlope1 = float(self.harmDevSlope1.get())
			
			nH = int(self.nH.get())
			stocf = float(self.stocf.get())
			
			inputFile2 = self.filelocation2.get()
			window2 =  self.w2_type.get()
			M2 = int(self.M2.get())
			N2 = int(self.N2.get())
			t2 = int(self.t2.get())
			minSineDur2 = float(self.minSineDur2.get())
			minf02 = int(self.minf02.get())
			maxf02 = int(self.maxf02.get())
			f0et2 = int(self.f0et2.get())
			harmDevSlope2 = float(self.harmDevSlope2.get())

			self.inputFile1, self.fs1, self.hfreq1, self.hmag1, self.stocEnv1, \
				self.inputFile2, self.hfreq2, self.hmag2, self.stocEnv2 = hM.analysis(inputFile1, window1, M1, N1, t1, \
				minSineDur1, nH, minf01, maxf01, f0et1, harmDevSlope1, stocf, inputFile2, window2, M2, N2, t2, minSineDur2, minf02, maxf02, f0et2, harmDevSlope2)

		except ValueError as errorMessage:
			tkMessageBox.showerror("Input values error", errorMessage)

	def transformation_synthesis(self):

		try:
			inputFile1 = self.inputFile1
			fs = self.fs1
			hfreq1 = self.hfreq1
			hmag1 = self.hmag1
			stocEnv1 = self.stocEnv1
			inputFile2 = self.inputFile2
			hfreq2 = self.hfreq2
			hmag2 = self.hmag2
			stocEnv2 = self.stocEnv2
			hfreqIntp = np.array(eval(self.hfreqIntp.get()))
			hmagIntp = np.array(eval(self.hmagIntp.get()))
			stocIntp = np.array(eval(self.stocIntp.get()))

			hM.transformation_synthesis(inputFile1, fs, hfreq1, hmag1, stocEnv1, inputFile2, hfreq2, hmag2, stocEnv2, hfreqIntp, hmagIntp, stocIntp)

		except ValueError as errorMessage:
			tkMessageBox.showerror("Input values error", errorMessage)

		except AttributeError:
			tkMessageBox.showerror("Analysis not computed", "First you must analyse the sound!")
