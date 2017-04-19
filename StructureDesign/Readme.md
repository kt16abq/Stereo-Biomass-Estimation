# Introduction
This script is established for Marine Science Department. It is also part of my graduate design. 

The default units are **mm** and **rad**.

# Run code
## Environment
The main file is named _go.m_. Before you run it, you have to check
1. Your Matlab Version is 2016b or higher
2. Default path (in MATLAB command) is the same as _go.m_
3. Manually add the toolbox in to Matlab path
4. There is a file named _inPara.mat_. It is the intrinc parameters of GoPro Cameras H4.
For new cameras, read http://www.vision.caltech.edu/bouguetj/calib_doc/ to learn how to get the parameters.

## Setting
Most of parameters are depend on the scale of sharks the distance from camera to sharks, and the distance between two cameras

The following steps tell you how to add these variables in to code.
1. In the same path of _go.m_, open file _distance.m_ in txt mode. You will find

  > function distance = distance
  
  >     distance = 800;
  
  > end
  
  here _distance_ just means the distance from sharks and people. You can change it in to your predicted value, or expected value.
  
2. Similarly, open file _L.m_ and _B.m_,
    respectively change them into the length of sharks and the longest possible distance between two cameras.

3. Make sure you saved the three files before you close them.

## Run!

In Matlab interface, find command window and type _go_, press Enter on keyboard.

Fisrtly you will see several figure named _optimal.._ coming one by one. Programme is looking for a global best values in your setting.

Then the figure stops moving, a counter starts running on command window. It have get your best value and starts drawing the error map.

At last two figures show up. The one named _Figure2_ shows the measurement error in every field of view, 
  and _Figure3_ explans when you set relative angle is not set precisely, how many extra error will be produced. 
  
# Get result
## Suggestion on stereo structure
1. To get the suggested distance between two cameras, or named _baseline_, 
  type _B_ on the command window and see result

2. To get the suggested relative angle between two cameras, or named _tr_, 
  type _x(2)_ on the command window and see result
  
## Suggestion on measurement
1. To get the suggested distance between shark and cameras, 
  type _x(1)_ on the command window and see result

2. To get the suggested the best position of sharks in your view, my suggestion is 
  let the position of sharks in two cameras be symmetric, and centre.
  
  
----------
By Frost
