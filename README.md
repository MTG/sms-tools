sms-tools
=========

<p>Spectral modeling analysis and synthesis tools written in python and C for sound and music applications.</p>

<p> In order to use all the software you have to have installed version 2.7 of python and install the following modules: iPython, Matplotlib, Numpy, Scipy, PyAudio, PySide, Cython.</p>

<p> In Ubuntu/debian all this is as easy as to type in a Terminal:

<p> <code> sudo apt-get install python2.7 </code></p>
<p> <code> sudo apt-get install python-pip python-dev python-scipy python-numpy python-matplotlib python-pyaudio python-pyside cython </code></p>


The basic analysis/synthesis models are in the directory software/models. The best way to execute the models is from inside iPython. For example to run the hpsModel you have to go to the software/models directory and write <code>run hpsModel</code> </p>

<p>There are examples of analysis/transformation/synthesis in the examples directory, all the sounds used in the examples are in the sounds directory, and there are class slides in the lectures directory to understand the concepts used.</p>

<p>In order to use the C functions, which will run most the code faster you need to compile basicFunctions_C. Onece Cython is installed in the Terminal go to the directory software/basicFunctions_C and write <code> python CompileModule.py build_ext --inplace </code> (don't bother if it appears a warning) </p>






