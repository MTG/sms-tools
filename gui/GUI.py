from Tkinter import *
from notebook import *
from dftModel_frame import *
from stft_frame import *

root = Tk( ) 
root.title('sms-tools GUI')
nb = notebook(root, TOP) # make a few diverse frames (panels), each using the NB as 'master': 

# uses the notebook's frame
f1 = Frame(nb( )) 
dft = DftModel_frame(f1)

f2 = Frame(nb( )) 
stft = Stft_frame(f2)
#b2 = Button(f2, text='Button 2') 
#b3 = Button(f2, text='Beep 2', command=Tk.bell) 
#b2.pack(fill=BOTH, expand=1) 
#b3.pack(fill=BOTH, expand=1) 

f3 = Frame(nb( )) 
f4 = Frame(nb( )) 
f5 = Frame(nb( )) 
f6 = Frame(nb( )) 
f7 = Frame(nb( )) 
f8 = Frame(nb( )) 
f9 = Frame(nb( )) 

nb.add_screen(f1, "DFT Model") 
nb.add_screen(f2, "STFT")
nb.add_screen(f3, "Sine Model") 
nb.add_screen(f4, "Harmonic Model") 
nb.add_screen(f5, "Stochastic Model") 
nb.add_screen(f6, "SPR Model") 
nb.add_screen(f7, "SPS Model") 
nb.add_screen(f8, "HPR Model") 
nb.add_screen(f9, "HPS Model") 

nb.display(f1)

root.mainloop( )