sms-tools
========= 

Sound analysis/synthesis tools for music applications written in python (with a bit of C) plus complementary lecture materials.

How to use
----------

In order to use these tools you have to install version 2.7.* of python and the following modules: ipython, numpy, matplotlib, scipy, and cython. 

In Ubuntu (which we strongly recommend) in order to install all these modules it is as simple as typing in the Terminal:

<code>$ sudo apt-get install python-dev ipython python-numpy python-matplotlib python-scipy cython</code>

In OSX (which has some problems and that we do not support) you install these modules by typing in the Terminal:

<code>$ pip install ipython numpy matplotlib scipy cython</code>

In OS X, you can use [homebrew](http://brew.sh/) and [pip](http://pip.readthedocs.org/en/latest/):

<code>$ brew install sdl sdl_image sdl_mixer sdl_ttf smpeg portmidi python</code>
<code>$ pip install ipython numpy matplotlib scipy hg+http://bitbucket.org/pygame/pygame cython</code>

then for using the tools, after downloading the whole package, you need to compile some C functions. For that you should go to the directory <code>software/models/utilFunctions_C</code> and type:</p>

<code>$ python compileModule.py build_ext --inplace </code>

The basic sound analysis/synthesis functions, or models, are in the directory <code>software/models</code> and there is a graphical interface and individual example functions in <code>software/models_interface</code>. To execute the models GUI you have to go to the directory <code>software/models_interface</code> and type: 

<code>$ python models_GUI.py </code>

To execute the transformations GUI that calls various sound transformation functions go to the directory <code>software/transformations_interface</code> and type: 

<code>$ python transformations_GUI.py </code>

To modify the existing code, or to create your own using some of the functions, we recommend to use the <code>workspace</code> directory. Typically you would copy a file from <code>software/models_interface</code> or from <code>software/transformations_interface</code> to that directory, modify the code, and execute it from there (you will have to change some of the paths inside the files). 


Content
-------

All the code is in the <code> software </code> directory, with subdirectories for the models, the transformations, and the interfaces. The lecture material is in the <code>lecture</code> directory and the sounds used for the examples and coming from <code>http://freesound.org</code> are in the <code>sounds</code> directory.

License
-------
All the software is distributed with the Affero GPL licence, and the lecture slides and sounds are distributed with the Creative Commons Attribution-Noncommercial-Share Alike license.

