sms-tools
========= 

<p>Spectral modeling analysis and synthesis tools for sound and music applications written in python
and C, plus complementary teaching material.</p>

<p>In order to use these software tools you have to install version 2.7 of python and the following modules: ipython, matplotlib, numpy, scipy, and pygame. 
</p>
<p>
In Ubuntu (which we strongly recommend) to install all the modules, plus the basic development tools, is as simple as typing on 
the Terminal:</p>
<p>
<code class="western">
$ sudo apt-get install python-dev python-setuptools build-essential ipython python-matplotlib python-numpy python-scipy python-pygame
</code>
</p>
<p>Some of the core functions are written in C and have to be compiled. For that,
you first have to install Cython, by typing on the Terminal: </p>
<p>
<code class="western">
$ easy_install cython
</code>
</p>
Once Cython is installed go to the directory software/models/utilFunctions_C and type:</p>
<p>
<code class="western">
$ python compileModule.py build_ext --inplace </code>
</p>

<p>The code for the basic analysis/synthesis models is in the
directory software/models. To use these models there is an interface and example functions in software/models_interface. All the sounds used in the examples are in the sounds directory.</p>

<p>To start we recommend to download the whole package, compile the C code with Cython and execute the GUI available in software/models_interface. To execute the GUI to call all the analysis/synthesis you have to use the Terminal, go to that directory and type: </p>
<code class="western">
$ python models_GUI.py </code>
</p>

<p>To execute the GUI to call various sound transformation go to that directory software/transformations_interface and type: </p>
<code class="western">
$ python transformations_GUI.py </code>
</p>

<p> To modify the existing code, or to create your own, we recommend to use the workspace directory. Typically you would copy a file from software/models_interface or from software/transformations_interface to that directory and execute the code from there (you will have to change some of the paths inside the files). </p>

<p>All this code is used in several classes that I teach. The slides
and demo code used in class are in the lectures directory.</p>