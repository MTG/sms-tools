sms-tools
========= 


Sound analysis/synthesis tools for music applications written in python (with a bit of C) plus complementary lecture materials.

How to use
----------

In order to use these tools you have to install version 2.7.* of python.

Simply run `python setup.py install` from the top-level directory to install sms_tools.

The basic sound analysis/synthesis functions, or models, are in the directory <code>sms_tools/models</code> and there is a graphical interface and individual example functions in <code>sms_tools/models_interface</code>. To execute the models GUI type: 

<code>$ bin/models_GUI.py </code>

To execute the transformations GUI that calls various sound transformation functions type: 

<code>$ bin/transformations_GUI.py </code>

Assignments
-----------

To begin working on an assignment, run <code>bin/startAssignment</code>. The program will prompt you to input the assignment you wish to work on, and then it will extract the proper zip file to a new subdirectory in your <code>workspace</code> directory.

To modify the existing code, or to create your own using some of the functions, we recommend to use the <code>workspace</code> directory. Typically you would copy a file from <code>sms_tools/models_interface</code> or from <code>sms_tools/transformations_interface</code> to that directory, modify the code, and execute it from there (you will have to change some of the paths inside the files). 


Content
-------

All the code is in the <code> sms_tools </code> directory, with subdirectories for the models, the transformations, and the interfaces. The lecture materials are in the <code>lectures</code> directory, the assignments related to the lectures in the  <code>assignments</code> directory, and the sounds used for the examples and coming from <code>http://freesound.org</code> are in the <code>sounds</code> directory.

License
-------
All the software is distributed with the Affero GPL license (http://www.gnu.org/licenses/agpl-3.0.en.html), the lecture slides are distributed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0) license (http://creativecommons.org/licenses/by-nc-sa/4.0/) and the sounds in this repository are released under Creative Commons Attribution 4.0 (CC BY 4.0) license (http://creativecommons.org/licenses/by/4.0/)

