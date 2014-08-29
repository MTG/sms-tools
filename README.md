sms-tools
========= 

Sound analysis and synthesis tools for music applications written in python and C, plus complementary teaching material.

How to use
----------

In order to use these software tools you have to install version 2.7 of python and the following modules: ipython, numpy, matplotlib, scipy, pygame, and cython. 

In Ubuntu (which we strongly recommend) to install all the modules is as simple as typing on 
the Terminal:

<code class="western">
$ sudo apt-get install python-dev ipython python-numpy python-matplotlib python-scipy python-pygame cython
</code>

Then to compile the C functions, go to the directory software/models/utilFunctions_C and type:</p>

<code class="western">
$ python compileModule.py build_ext --inplace </code>

The basic sound analysis/synthesis functions, or models, are in the directory <code>software/models</code>. To use these them there is a graphical interface and individual example functions in <code>software/models_interface</code>. All the sounds used are in the <code>sounds</code> directory.

<p>To start we recommend to download the whole package, compile the C functions (as described above) and execute the GUI available in software/models_interface. To execute the GUI you have to use the Terminal, going to the directory <code>software/models_interface</code> and typing: 

<code class="western">
$ python models_GUI.py </code>

To execute the GUI that calls various sound transformation functions go to the directory <code>software/transformations_interface</code> and type: 

<code class="western">
$ python transformations_GUI.py </code>

To modify the existing code, or to create your own, we recommend to use the <code>workspace</code> directory. Typically you would copy a file from <code>software/models_interface</code> or from software/transformations_interface to that directory, modify the code, and execute the it from there (you will have to change some of the paths inside the files). 



Content
-------

This repository includes all the software tools in the <code> software </code> directory, the lecture material in the <code>lecture</code> directory and the sounds used for the examples and comming from <code>http://freesound.org</code> in the <code>sounds</code> directory.

License
-------
All the software is distributed with the Affero GP licence, and the slides of the lectures and sounds with the Creative Commons Attribution-Noncommercial-Share Alike license.

