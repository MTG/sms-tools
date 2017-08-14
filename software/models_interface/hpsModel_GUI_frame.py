# GUI frame for the hprModel_function.py

try:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter 
    import tkFileDialog, tkMessageBox
except ImportError:
    # for Python3
    from tkinter import *   ## notice lowercase 't' in tkinter here
    from tkinter import filedialog as tkFileDialog
    from tkinter import messagebox as tkMessageBox
import sys, os
from scipy.io.wavfile import read
import hpsModel_function
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
 
class HpsModel_frame:
  
	def __init__(self, parent):  
		 
		self.parent = parent        
		self.initUI()

	def initUI(self):

		choose_label = "Input file (.wav, mono and 44100 sampling rate):"
		Label(self.parent, text=choose_label).grid(row=0, column=0, sticky=W, padx=5, pady=(10,2))
 
		#TEXTBOX TO PRINT PATH OF THE SOUND FILE
		self.filelocation = Entry(self.parent)
		self.filelocation.focus_set()
		self.filelocation["width"] = 25
		self.filelocation.grid(row=1,column=0, sticky=W, padx=10)
		self.filelocation.delete(0, END)
		self.filelocation.insert(0, '../../sounds/sax-phrase-short.wav')

		#BUTTON TO BROWSE SOUND FILE
		self.open_file = Button(self.parent, text="Browse...", command=self.browse_file) #see: def browse_file(self)
		self.open_file.grid(row=1, column=0, sticky=W, padx=(220, 6)) #put it beside the filelocation textbox
 
		#BUTTON TO PREVIEW SOUND FILE
		self.preview = Button(self.parent, text=">", command=lambda:UF.wavplay(self.filelocation.get()), bg="gray30", fg="white")
		self.preview.grid(row=1, column=0, sticky=W, padx=(306,6))

		## HARMONIC MODEL

		#ANALYSIS WINDOW TYPE
		wtype_label = "Window type:"
		Label(self.parent, text=wtype_label).grid(row=2, column=0, sticky=W, padx=5, pady=(10,2))
		self.w_type = StringVar()
		self.w_type.set("blackman") # initial value
		window_option = OptionMenu(self.parent, self.w_type, "rectangular", "hanning", "hamming", "blackman", "blackmanharris")
		window_option.grid(row=2, column=0, sticky=W, padx=(95,5), pady=(10,2))

		#WINDOW SIZE
		M_label = "Window size (M):"
		Label(self.parent, text=M_label).grid(row=4, column=0, sticky=W, padx=5, pady=(10,2))
		self.M = Entry(self.parent, justify=CENTER)
		self.M["width"] = 5
		self.M.grid(row=4,column=0, sticky=W, padx=(115,5), pady=(10,2))
		self.M.delete(0, END)
		self.M.insert(0, "601")

		#FFT SIZE
		N_label = "FFT size (N) (power of two bigger than M):"
		Label(self.parent, text=N_label).grid(row=5, column=0, sticky=W, padx=5, pady=(10,2))
		self.N = Entry(self.parent, justify=CENTER)
		self.N["width"] = 5
		self.N.grid(row=5,column=0, sticky=W, padx=(270,5), pady=(10,2))
		self.N.delete(0, END)
		self.N.insert(0, "1024")

		#THRESHOLD MAGNITUDE
		t_label = "Magnitude threshold (t) (in dB):"
		Label(self.parent, text=t_label).grid(row=6, column=0, sticky=W, padx=5, pady=(10,2))
		self.t = Entry(self.parent, justify=CENTER)
		self.t["width"] = 5
		self.t.grid(row=6, column=0, sticky=W, padx=(205,5), pady=(10,2))
		self.t.delete(0, END)
		self.t.insert(0, "-100")

		#MIN DURATION SINUSOIDAL TRACKS
		minSineDur_label = "Minimum duration of sinusoidal tracks:"
		Label(self.parent, text=minSineDur_label).grid(row=7, column=0, sticky=W, padx=5, pady=(10,2))
		self.minSineDur = Entry(self.parent, justify=CENTER)
		self.minSineDur["width"] = 5
		self.minSineDur.grid(row=7, column=0, sticky=W, padx=(250,5), pady=(10,2))
		self.minSineDur.delete(0, END)
		self.minSineDur.insert(0, "0.1")

		#MAX NUMBER OF HARMONICS
		nH_label = "Maximum number of harmonics:"
		Label(self.parent, text=nH_label).grid(row=8, column=0, sticky=W, padx=5, pady=(10,2))
		self.nH = Entry(self.parent, justify=CENTER)
		self.nH["width"] = 5
		self.nH.grid(row=8, column=0, sticky=W, padx=(215,5), pady=(10,2))
		self.nH.delete(0, END)
		self.nH.insert(0, "100")

		#MIN FUNDAMENTAL FREQUENCY
		minf0_label = "Minimum fundamental frequency:"
		Label(self.parent, text=minf0_label).grid(row=9, column=0, sticky=W, padx=5, pady=(10,2))
		self.minf0 = Entry(self.parent, justify=CENTER)
		self.minf0["width"] = 5
		self.minf0.grid(row=9, column=0, sticky=W, padx=(220,5), pady=(10,2))
		self.minf0.delete(0, END)
		self.minf0.insert(0, "350")

		#MAX FUNDAMENTAL FREQUENCY
		maxf0_label = "Maximum fundamental frequency:"
		Label(self.parent, text=maxf0_label).grid(row=10, column=0, sticky=W, padx=5, pady=(10,2))
		self.maxf0 = Entry(self.parent, justify=CENTER)
		self.maxf0["width"] = 5
		self.maxf0.grid(row=10, column=0, sticky=W, padx=(220,5), pady=(10,2))
		self.maxf0.delete(0, END)
		self.maxf0.insert(0, "700")

		#MAX ERROR ACCEPTED
		f0et_label = "Maximum error in f0 detection algorithm:"
		Label(self.parent, text=f0et_label).grid(row=11, column=0, sticky=W, padx=5, pady=(10,2))
		self.f0et = Entry(self.parent, justify=CENTER)
		self.f0et["width"] = 5
		self.f0et.grid(row=11, column=0, sticky=W, padx=(265,5), pady=(10,2))
		self.f0et.delete(0, END)
		self.f0et.insert(0, "5")

		#ALLOWED DEVIATION OF HARMONIC TRACKS
		harmDevSlope_label = "Max frequency deviation in harmonic tracks:"
		Label(self.parent, text=harmDevSlope_label).grid(row=12, column=0, sticky=W, padx=5, pady=(10,2))
		self.harmDevSlope = Entry(self.parent, justify=CENTER)
		self.harmDevSlope["width"] = 5
		self.harmDevSlope.grid(row=12, column=0, sticky=W, padx=(285,5), pady=(10,2))
		self.harmDevSlope.delete(0, END)
		self.harmDevSlope.insert(0, "0.01")

		#DECIMATION FACTOR
		stocf_label = "Stochastic approximation factor:"
		Label(self.parent, text=stocf_label).grid(row=13, column=0, sticky=W, padx=5, pady=(10,2))
		self.stocf = Entry(self.parent, justify=CENTER)
		self.stocf["width"] = 5
		self.stocf.grid(row=13, column=0, sticky=W, padx=(210,5), pady=(10,2))
		self.stocf.delete(0, END)
		self.stocf.insert(0, "0.2")

		#BUTTON TO COMPUTE EVERYTHING
		self.compute = Button(self.parent, text="Compute", command=self.compute_model, bg="dark red", fg="white")
		self.compute.grid(row=14, column=0, padx=5, pady=(10,2), sticky=W)

		#BUTTON TO PLAY SINE OUTPUT
		output_label = "Sinusoidal:"
		Label(self.parent, text=output_label).grid(row=15, column=0, sticky=W, padx=5, pady=(10,0))
		self.output = Button(self.parent, text=">", command=lambda:UF.wavplay('output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_hpsModel_sines.wav'), bg="gray30", fg="white")
		self.output.grid(row=15, column=0, padx=(80,5), pady=(10,0), sticky=W)

		#BUTTON TO PLAY STOCHASTIC OUTPUT
		output_label = "Stochastic:"
		Label(self.parent, text=output_label).grid(row=16, column=0, sticky=W, padx=5, pady=(5,0))
		self.output = Button(self.parent, text=">", command=lambda:UF.wavplay('output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_hpsModel_stochastic.wav'), bg="gray30", fg="white")
		self.output.grid(row=16, column=0, padx=(80,5), pady=(5,0), sticky=W)

		#BUTTON TO PLAY OUTPUT
		output_label = "Output:"
		Label(self.parent, text=output_label).grid(row=17, column=0, sticky=W, padx=5, pady=(5,15))
		self.output = Button(self.parent, text=">", command=lambda:UF.wavplay('output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_hpsModel.wav'), bg="gray30", fg="white")
		self.output.grid(row=17, column=0, padx=(80,5), pady=(5,15), sticky=W)


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

	def compute_model(self):
		
		try:
			inputFile = self.filelocation.get()
			window = self.w_type.get()
			M = int(self.M.get())
			N = int(self.N.get())
			t = int(self.t.get())
			minSineDur = float(self.minSineDur.get())
			nH = int(self.nH.get())
			minf0 = int(self.minf0.get())
			maxf0 = int(self.maxf0.get())
			f0et = int(self.f0et.get())
			harmDevSlope = float(self.harmDevSlope.get())
			stocf = float(self.stocf.get())

			hpsModel_function.main(inputFile, window, M, N, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope, stocf)

		except ValueError as errorMessage:
			tkMessageBox.showerror("Input values error", errorMessage)
