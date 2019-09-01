This directory includes a set of exercises to practice the concepts covered in the lectures, thus helping understand most of the sms-tools software. 

All the exercises are given as Jupyter notebooks. For documentation go to: https://jupyter.readthedocs.io/en/latest/.

Setting up sms-tools

In these exercises you will be using many functions of sms-tools. It is therefore important to setup sms-tools correctly and make sure it works. We recommend to clone the repository using git clone. Since the repository evolves over time and undergoes changes, it is easier to keep the code synchronized with the latest version on github through git. For the purposes of these exercises, you just need to use the clone and pull commands of git.

Installation

To start working with the software of sms-tools you need to install some dependencies. Read the instructions in the README file: https://github.com/MTG/sms-tools/blob/master/README.md.

Working directory for exercises

In sms-tools there is a folder called workspace. We will use this folder as our working directory for all the programming exercises. You should copy the notebook of the exercise you want to do to the workspace directory. 

Sound files

Most exercises require the use of sound files. You can use your own, making sure that they are in the right format, or you can use the ones in the folder sounds of sms-tools, which are all in the required format. To facilitate the work, we have restricted the formats of sound files to use. You should always use wav audio files that are mono (one channel), sampled at 44100 Hz and stored as 16 bit integers. 

Within the code, we use floating point arrays to store the sound samples, thus the samples in the audio files are converted to floating point values with a range from -1 to 1. The wavread function from the utilFunctions module in the sms-tools reads a wav file and returns a floating-point array with the sample values normalized to the range -1 to 1, which is what we want. 

Testing you code

After you complete a part of an exercise, make sure you run and test the code. All the programming exercises questions require that you write python functions. Each question will give a basic function template, specifying an input, or a type of input, and what output the function has to return. 

In the explanation of each exercise question we describe some example test cases that you can use to run and test your code and make sure that it gives meaningful results. At times, your answers (output of your code) may differ from the reference output provided in the question description.  Ignore these differences if they are beyond the third decimal place as they are insignificant and will not be penalized.