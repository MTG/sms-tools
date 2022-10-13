import matplotlib
matplotlib.use('TkAgg')

from notebook import *  # window with tabs
from dftModel_GUI_frame import *
from stft_GUI_frame import *
from sineModel_GUI_frame import *
from harmonicModel_GUI_frame import *
from stochasticModel_GUI_frame import *
from sprModel_GUI_frame import *
from spsModel_GUI_frame import *
from hprModel_GUI_frame import *
from hpsModel_GUI_frame import *

root = Tk()
root.title('sms-tools models GUI')
nb = notebook(root, TOP)  # make a few diverse frames (panels), each using the NB as 'master':

# uses the notebook's frame
f1 = Frame(nb())
dft = DftModel_frame(f1)

f2 = Frame(nb())
stft = Stft_frame(f2)

f3 = Frame(nb())
sine = SineModel_frame(f3)

f4 = Frame(nb())
harmonic = HarmonicModel_frame(f4)

f5 = Frame(nb())
stochastic = StochasticModel_frame(f5)

f6 = Frame(nb())
spr = SprModel_frame(f6)

f7 = Frame(nb())
sps = SpsModel_frame(f7)

f8 = Frame(nb())
hpr = HprModel_frame(f8)

f9 = Frame(nb())
hps = HpsModel_frame(f9)

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

root.geometry('+0+0')
root.mainloop()
