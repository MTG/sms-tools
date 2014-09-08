sms-tools
========= 

Sound analysis/synthesis tools for music applications written in python (with a bit of C) plus complementary lecture materials.

How to use
----------

In order to use these tools you have to install version 2.7.* of python and the following modules: ipython, numpy, matplotlib, scipy, pygame, and cython. 

In Ubuntu (which we strongly recommend) in order to install all these modules it is as simple as typing in the Terminal:

```sh
$ sudo apt-get install python-dev ipython python-numpy python-matplotlib python-scipy python-pygame cython
```

Windows users should install [python(x,y)](code.google.com/p/pythonxy/) to setup a python environment. Remember to check `Cython` and `MinGW` in pythonxy installer.

Minimized versions of these packages are attached in *Downloads* section so you don't have to install them by hand. Just follow the procedure:

1. Unpack the `.7z` archives into `C:\MSYS` and `C:\Python27`.
2. Goto *Advanced System Settings -> Environment Variables*, add `;C:\MSYS\bin;C:\MSYS\local\bin;C:\Python27;C:\Python27\DLLs;C:\Python27\Scripts;` to `PATH` variable. 
3. Open a `cmd` prompt and type: `sh`

then for using the tools, after downloading the whole package, you need to compile some C functions. For that you should go to the directory `software/models/utilFunctions_C` and type:

```sh
$ python compileModule.py build_ext --inplace
```

The basic sound analysis/synthesis functions, or models, are in the directory `software/models` and there is a graphical interface and individual example functions in `software/models_interface`. To execute the models GUI you have to go to the directory `software/models_interface` and type: 

```sh
$ python models_GUI.py
```

To execute the transformations GUI that calls various sound transformation functions go to the directory `software/transformations_interface` and type: 

```sh
$ python transformations_GUI.py
```

To modify the existing code, or to create your own using some of the functions, we recommend to use the `workspace` directory. Typically you would copy a file from `software/models_interface` or from `software/transformations_interface` to that directory, modify the code, and execute it from there (you will have to change some of the paths inside the files). 


Content
-------

All the code is in the `software` directory, with subdirectories for the models, the transformations, and the interfaces. The lecture material is in the `lecture` directory and the sounds used for the examples and coming from `http://freesound.org` are in the `sounds` directory.

Downloads
-------

Minimized package of Pythonxy(including numpy, scipy, matplotlib, pygame, cython and pyqt4 core).

[Download from Mega](https://mega.co.nz/#!zRQSSIZZ!XNeLUpcJs6ZLM3eX-lr4v8Mj1xsgFBpexwYJwcWSb1E), 56MiB

Minimized package of MSYS(including core-utils, GNU Make, gcc, and g++).

[Download from Mega](https://mega.co.nz/#!6VBgybCA!Tf4-9Jtrdrnc4qGpG0Be5-71-FM5BZQs0laOzbwLPKk), 28MiB

For Chinese users,

[Download from BaiduYun](http://pan.baidu.com/s/1eQpbsWi), can be faster accessed.

License
-------
All the software is distributed with the Affero GPL licence, and the lecture slides and sounds are distributed with the Creative Commons Attribution-Noncommercial-Share Alike license.

