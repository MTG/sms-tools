sms-tools
========= 

<p>Spectral modeling analysis and synthesis tools for sound and music applications written in python
and C, plus complementary teaching material.</p>

How to use
----------

<p>In order to use these software tools you have to install version 2.7 of python and the following modules: ipython, numpy, matplotlib, scipy, pygame, and cython. 
</p>
<p>
In Ubuntu (which we strongly recommend) to install all the modules is as simple as typing on 
the Terminal:</p>
<p>
<code class="western">
$ sudo apt-get install python-dev ipython python-numpy python-matplotlib python-scipy python-pygame cython
</code>
</p>
Then to compile the C functions, go to the directory software/models/utilFunctions_C and type:</p>
<p>
<code class="western">
$ python compileModule.py build_ext --inplace </code>
</p>

<p>The basic sound analysis/synthesis functions, or models, are in the
directory software/models. To use these them there is a graphical interface and individual example functions in software/models_interface. All the sounds used are in the sounds directory.</p>

<p>To start we recommend to download the whole package, compile the C functions (as described above) and execute the GUI available in software/models_interface. To execute the GUI you have to use the Terminal, going to the directory software/models_interface and typing: </p>
<code class="western">
$ python models_GUI.py </code>
</p>

<p>To execute the GUI that calls various sound transformation functions go to the directory software/transformations_interface and type: </p>
<code class="western">
$ python transformations_GUI.py </code>
</p>

<p> To modify the existing code, or to create your own, we recommend to use the workspace directory. Typically you would copy a file from software/models_interface or from software/transformations_interface to that directory, modify the code, and execute the it from there (you will have to change some of the paths inside the files). </p>

<p>All this code is used in several classes that I teach. The slides
and demo code used in class are in the lectures directory.</p>


Content
-------

This repository include all the software tools in the <code> software </code> directory, the lecture material in the <code>lecture</code> directory and the sounds use for the examples and comming from <code>http://freesound.org</code> in the <code>sounds</code> directory.

License
-------
All the software is distributed with the Affero GP licence, the slides of the lectures and sounds with the Creative Commons Attribution-Noncommercial-Share Alike license.

