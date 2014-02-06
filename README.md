sms-tools
=========

<p>Spectral modeling analysis and synthesis tools written in python and C for sound and music applications.</p>


<p> In order to use them you have to install: iPython, Matplotlib, Numpy, Scipy, PyAudio, PySide, Cython.</p>
</ul>

The basic analysis/synthesis models are in the directory software/models. The best way to execute the models is from inside iPython. For example to run the hpsModel you have to go to the software/models directory and type:

<p> <code>run hpsModel</code> </p>

<p>There are examples of analysis/transformation/synthesis in the examples directory, all the sounds used in the examples are in the sounds directory, and there are class slides in the lectures directory to understand the concepts used.</p>

<p>In order to use the C functions and thus run the code faster you need to compile UtilityFunctions by:</p>

<ol>
<li>Install cython (<code>easy_install cython</code>) </li>
<li>Download all the files from the UtilityFunctions folder in your directory with all the other codes (hpsGui.py, sps.py...) </li>
<li>Go to the directory with the terminal and write <code> python CompileModule.py build_ext --inplace </code> (don't bother if it appears a warning) </li>

</ol>





