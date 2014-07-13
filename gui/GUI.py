from Tkinter import *
from notebook import *
from dftModel_frame import *
from stft_frame import *
from sineModel_frame import *

root = Tk( ) 
root.title('sms-tools GUI')
nb = notebook(root, TOP) # make a few diverse frames (panels), each using the NB as 'master': 

# uses the notebook's frame
f1 = Frame(nb( )) 
dft = DftModel_frame(f1)

f2 = Frame(nb( )) 
stft = Stft_frame(f2)

f3 = Frame(nb( )) 
sine = SineModel_frame(f3)

f4 = Frame(nb( )) 
f5 = Frame(nb( )) 
f6 = Frame(nb( )) 
f7 = Frame(nb( )) 
f8 = Frame(nb( )) 
f9 = Frame(nb( )) 

nb.add_screen(f1, "DFT") 
nb.add_screen(f2, "STFT")
nb.add_screen(f3, "Sine") 
nb.add_screen(f4, "Harmonic") 
nb.add_screen(f5, "Stochastic") 
nb.add_screen(f6, "SPR") 
nb.add_screen(f7, "SPS") 
nb.add_screen(f8, "HPR") 
nb.add_screen(f9, "HPS") 

nb.display(f1)

root.mainloop( )