sms-tools
=========

<p>Spectral modeling analysis and synthesis tools written in python and C for sound and music applications, plus some complementary material.</p>

<p> In order to use all the software you have to install version 2.7 of python and the following modules: iPython, Matplotlib, Numpy, Scipy, PyAudio, PySide, Cython. Some of the code also requires to install the <a href="http://essentia.upf.edu/"> Essentia library</a>.  </p>

<p>For information on how to install python and the needed modules we refer to the <a href="http://essentia.upf.edu/documentation/installing.html"> documentation</a> to install Essentia, which covers all this.</p>

<p>The code for the basic analysis/synthesis models is in the directory software/models. You can run the code from inside iPython, for example by typing <code>run hpsModel.py</code>, or from the Terminal, for example going to the software/models directory and typing <code>python hpsModel.py</code> </p>

<p>There are examples of analysis/transformation/synthesis in the examples directory. All the sounds used in the examples are in the sounds directory.</p>

<p>In order to use the C functions, which will run most the code faster, you need to compile basicFunctions_C. Once Cython is installed, in the Terminal go to the directory software/utilFunctions_C and write <code> python compileModule.py build_ext --inplace </code> (don't bother if it appears a warning) </p>

<p>All this code is used in several classes that I teach. The slides and demo code I use in class is available in the lectures directory.</p>






