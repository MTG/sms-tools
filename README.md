sms-tools
========= 


Sound analysis/synthesis tools for music applications written in python (with a bit of C) plus complementary teaching materials.

How to use
----------

In order to use these tools you have to install python 3.* (recommended 3.10) and the following modules: <code>ipython</code>, <code>numpy</code>, <code>matplotlib</code>, <code>scipy</code>, and <code>cython</code>. 

In Ubuntu you can install these modules by typing in the Terminal:

<code>$ sudo apt-get install python-dev ipython python-numpy python-matplotlib python-scipy cython</code>

In OSX you can install these modules by typing in the Terminal:

<code>$ pip install ipython numpy matplotlib scipy cython</code>

then, to use the tools, after downloading the whole package, you need to compile some C functions. For that you should go to the directory <code>software/models/utilFunctions_C</code> and type:</p>

<code>$ python compileModule.py build_ext --inplace </code>

The basic sound analysis/synthesis functions, or models, are in the directory <code>software/models</code> and there is a graphical interface and individual example functions in <code>software/models_interface</code>. To execute the models GUI you have to go to the directory <code>software/models_interface</code> and type: 

<code>$ python models_GUI.py </code>

To execute the transformations GUI that calls various sound transformation functions go to the directory <code>software/transformations_interface</code> and type: 

<code>$ python transformations_GUI.py </code>

To modify the existing code, or to create your own using some of the functions, we recommend to use the <code>workspace</code> directory. Typically you would copy a file from <code>software/models_interface</code> or from <code>software/transformations_interface</code> to that directory, modify the code, and execute it from there (you will have to change some of the paths inside the files). 

Jupyter Notebooks
-------
A number of teaching exercises are available as jupyter notebooks in the <code>notebook</code> directory. To do them you need to install Jupyter Notebook according to its instructions https://jupyter.org/install

Start up jupyter notebook by typing in the Terminal.

<code>$ jupyter notebook</code> 

It will open a web browser, from which you can open the <code>notebook</code> directory.

Content
-------

All the code is in the <code> software </code> directory, with subdirectories for the models, the transformations, and the interfaces. The lecture materials are in the <code>lectures</code> directory, the exercises related to the lectures are in the  <code>notebook</code> directory, and the sounds used for the examples and coming from http://freesound.org are in the <code>sounds</code> directory.

License
-------
All the software is distributed with the Affero GPL license (http://www.gnu.org/licenses/agpl-3.0.en.html), the lecture slides are distributed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0) license (http://creativecommons.org/licenses/by-nc-sa/4.0/) and the sounds in this repository are released under Creative Commons Attribution 4.0 (CC BY 4.0) license (http://creativecommons.org/licenses/by/4.0/)

