import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


timeScale = np.array([.01, .0, .03, .03, .335, .4, .355, .42, .671, .8, .691, .82, .858, 1.2, .878, 1.22, 1.185, 1.6, 1.205, 1.62, 1.497, 2.0, 1.517, 2.02, 1.686, 2.4, 1.706, 2.42, 1.978, 2.8])          

plt.figure(1, figsize=(9, 6))

plt.plot(timeScale[::2], timeScale[1::2], lw=1.5)
plt.autoscale(tight=True)
# plt.axis([0,x.size/float(fs),min(x),max(x)])
plt.title('time scaling envelope')  
plt.xlabel('input time')  
plt.ylabel('output time')                       
plt.autoscale(tight=True)
plt.savefig('timescale.png')
plt.show()

