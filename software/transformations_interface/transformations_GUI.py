from Tkinter import *
from notebook import *   # window with tabs

from harmonicTransformations_GUI_frame import *
#from stochasticTransformations_GUI_frame import *

root = Tk( ) 
root.title('sms-tools GUI')
nb = notebook(root, TOP) # make a few diverse frames (panels), each using the NB as 'master': 

# uses the notebook's frame
f1 = Frame(nb( )) 
stochastic = HarmonicTransformations_frame(f1)

f2 = Frame(nb( )) 
#dft = DftModel_frame(f2)

f3 = Frame(nb( )) 
#sine = SineModel_frame(f3)

f4 = Frame(nb( )) 
f5 = Frame(nb( )) 

nb.add_screen(f1, "Harmonic") 
nb.add_screen(f2, "Harmonic2")
nb.add_screen(f3, "HPS")
nb.add_screen(f4, "STFTMorph") 
nb.add_screen(f5, "Stochastic") 

nb.display(f1)

root.geometry('+0+0')
root.mainloop( )