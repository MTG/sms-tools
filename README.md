sms-tools
=========

spectral modelling analysis and synthesis tools for sound and music applications

<h3>Requirements:</h3>
<ul>
<li>iPython</li>
<li>Matplotlib</li>
<li>Numpy</li>
<li>Scipy</li>
<li>PyAudio</li>
<li>PySide</li>
<li>Cython</li>
</ul>

<hr>

All the basic models for analysis and synthesis are in software/models

<h3>How to execute the software:</h3>

<p> In your terminal: <code>python hpsModel.py</code> </p>
<p> Inside iPython --pylab: <code>run hpsModel</code> </p>

<h3>How to compile UtilityFunctions:</h3>

<ol>
<li>Install cython (<code>easy_install cython</code>) </li>
<li>Download all the files from the UtilityFunctions folder in your directory with all the other codes (hpsGui.py, sps.py...) </li>
<li>Go to the directory with the terminal and write <code> python CompileModule.py build_ext --inplace </code> (don't bother if it appears a warning) </li>

</ol>





