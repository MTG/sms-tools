# GUI frame for the dftModel_function.py

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
import dftModel_function
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF

class DftModel_frame:
  
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
		self.filelocation.insert(0, '../../sounds/piano.wav')

		#BUTTON TO BROWSE SOUND FILE
		self.open_file = Button(self.parent, text="Browse...", command=self.browse_file) #see: def browse_file(self)
		self.open_file.grid(row=1, column=0, sticky=W, padx=(220, 6)) #put it beside the filelocation textbox
 
		#BUTTON TO PREVIEW SOUND FILE
		self.preview = Button(self.parent, text=">", command=lambda:UF.wavplay(self.filelocation.get()), bg="gray30", fg="white")
		self.preview.grid(row=1, column=0, sticky=W, padx=(306,6))

		## DFT MODEL

		#ANALYSIS WINDOW TYPE
		wtype_label = "Window type:"
		Label(self.parent, text=wtype_label).grid(row=2, column=0, sticky=W, padx=5, pady=(10,2))
		self.w_type = StringVar()
		self.w_type.set("blackman") # initial value
		window_option = OptionMenu(self.parent, self.w_type, "rectangular", "hanning", "hamming", "blackman", "blackmanharris")
		window_option.grid(row=2, column=0, sticky=W, padx=(95,5), pady=(10,2))

		#WINDOW SIZE
		M_label = "Window size (M):"
		Label(self.parent, text=M_label).grid(row=3, column=0, sticky=W, padx=5, pady=(10,2))
		self.M = Entry(self.parent, justify=CENTER)
		self.M["width"] = 5
		self.M.grid(row=3,column=0, sticky=W, padx=(115,5), pady=(10,2))
		self.M.delete(0, END)
		self.M.insert(0, "511")

		#FFT SIZE
		N_label = "FFT size (N) (power of two bigger than M):"
		Label(self.parent, text=N_label).grid(row=4, column=0, sticky=W, padx=5, pady=(10,2))
		self.N = Entry(self.parent, justify=CENTER)
		self.N["width"] = 5
		self.N.grid(row=4,column=0, sticky=W, padx=(270,5), pady=(10,2))
		self.N.delete(0, END)
		self.N.insert(0, "1024")

		#TIME TO START ANALYSIS
		time_label = "Time in sound (in seconds):"
		Label(self.parent, text=time_label).grid(row=5, column=0, sticky=W, padx=5, pady=(10,2))
		self.time = Entry(self.parent, justify=CENTER)
		self.time["width"] = 5
		self.time.grid(row=5, column=0, sticky=W, padx=(180,5), pady=(10,2))
		self.time.delete(0, END)
		self.time.insert(0, ".2")

		#BUTTON TO COMPUTE EVERYTHING
		self.compute = Button(self.parent, text="Compute", command=self.compute_model, bg="dark red", fg="white")
		self.compute.grid(row=6, column=0, padx=5, pady=(10,15), sticky=W)

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
			time = float(self.time.get())
			
			dftModel_function.main(inputFile, window, M, N, time)

		except ValueError as errorMessage:
			tkMessageBox.showerror("Input values error",errorMessage)
			
