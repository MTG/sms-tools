from Tkinter import *
import tkFileDialog, tkMessageBox
import sys, os
import pygame
from scipy.io.wavfile import read

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../examples/'))
import harmonicModel_example
 
class Harmonic_frame:
  
    def __init__(self, parent):  
         
        self.parent = parent        
        self.initUI()
        pygame.init()

    def initUI(self):

        choose_label = "Choose an input audio file .wav (monophonic with sampling rate of 44100 Hz):"
        Label(self.parent, text=choose_label).grid(row=0, column=0, sticky=W, padx=5, pady=(10,2))
 
        #TEXTBOX TO PRINT PATH OF THE SOUND FILE
        self.filelocation = Entry(self.parent)
        self.filelocation.focus_set()
        self.filelocation["width"] = 60
        self.filelocation.grid(row=1,column=0, columnspan=3, sticky=W+E, padx=10)
        self.filelocation.delete(0, END)
        self.filelocation.insert(0, '../sounds/piano.wav')

        #BUTTON TO BROWSE SOUND FILE
        self.open_file = Button(self.parent, text="Browse...", command=self.browse_file) #see: def browse_file(self)
        self.open_file.grid(row=1, column=3, padx=2) #put it beside the filelocation textbox
 
        #BUTTON TO PREVIEW SOUND FILE
        self.preview = Button(self.parent, text=">", command=self.preview_sound, bg="gray30", fg="white")
        self.preview.grid(row=1, column=4, padx=(2,6))

        ## HARMONIC MODEL

        #ANALYSIS WINDOW TYPE
        wtype_label = "Analysis window type (choice of rectangular, hanning, hamming, blackman, blackmanharris):"
        Label(self.parent, text=wtype_label).grid(row=2, column=0, sticky=W, padx=5, pady=(10,2))
        self.w_type = StringVar()
        self.w_type.set("hamming") # initial value
        window_option = OptionMenu(self.parent, self.w_type, "rectangular", "hanning", "hamming", "blackman", "blackmanharris")
        window_option.grid(row=3, column=0, sticky=W, padx=10)

        #WINDOW SIZE
        wsize_label = "Analysis window size M (odd integer value, e.g. 511):"
        Label(self.parent, text=wsize_label).grid(row=4, column=0, sticky=W, padx=5, pady=(10,2))
        self.M_entry = Entry(self.parent, justify=CENTER)
        self.M_entry["width"] = 8
        self.M_entry.grid(row=5,column=0, sticky=W, padx=10)
        self.M_entry.delete(0, END)
        self.M_entry.insert(0, "511")

        #FFT SIZE
        fft_label = "FFT size N (power of two, bigger than M, e.g. 1024):"
        Label(self.parent, text=fft_label).grid(row=6, column=0, sticky=W, padx=5, pady=(10,2))
        self.N_entry = Entry(self.parent, justify=CENTER)
        self.N_entry["width"] = 8
        self.N_entry.grid(row=7,column=0, sticky=W, padx=10)
        self.N_entry.delete(0, END)
        self.N_entry.insert(0, "1024")

        #TIME TO START ANALYSIS
        time_label = "Time to start analysis (in seconds):"
        Label(self.parent, text=time_label).grid(row=8, column=0, sticky=W, padx=5, pady=(10,2))
        self.time_entry = Entry(self.parent, justify=CENTER)
        self.time_entry["width"] = 8
        self.time_entry.grid(row=9,column=0, sticky=W, padx=10)
        self.time_entry.delete(0, END)
        self.time_entry.insert(0, ".2")

        #BUTTON TO COMPUTE EVERYTHING
        self.compute = Button(self.parent, text="Compute", command=self.compute_model, bg="dark red", fg="white")
        self.compute.grid(row=10, column=0, padx=5, pady=(20,2), sticky=W)

        # define options for opening file
        self.file_opt = options = {}
        options['defaultextension'] = '.wav'
        options['filetypes'] = [('All files', '.*'), ('Wav files', '.wav')]
        options['initialdir'] = '../sounds/'
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

    def compute_model(self):
        
        try:
            M = int(self.M_entry.get())
            N = int(self.N_entry.get())
            time = float(self.time_entry.get())
            dftModel_example.main(self.filelocation.get(), self.w_type.get(), M, N, time)

        except ValueError:
            tkMessageBox.showerror("Input values error", "Some parameters are incorrect")