import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


freqScaling = np.array([0, .8, 1, 1.2]) 
plt.figure(1, figsize=(9, 6))

plt.plot(freqScaling[::2], freqScaling[1::2], lw=1.5)
plt.autoscale(tight=True)
# plt.axis([0,x.size/float(fs),min(x),max(x)])
plt.title('frequency scaling envelope')  
plt.xlabel('time')  
plt.ylabel('scaling factor')                       
plt.autoscale(tight=True)
plt.savefig('freqscale.png')
plt.show()

