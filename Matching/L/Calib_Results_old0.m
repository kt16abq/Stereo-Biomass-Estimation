% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly executed under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 1160.816243241908500 ; 1240.647835667349500 ];

%-- Principal point:
cc = [ 1007.426237386244900 ; 551.979639768504060 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ -0.238273996954656 ; 0.060990261010769 ; 0.004822999693218 ; -0.003123636851706 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 177.440422029747050 ; 187.407792184591300 ];

%-- Principal point uncertainty:
cc_error = [ 44.607370131267004 ; 76.758391013468170 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.060096625110885 ; 0.034778506234275 ; 0.007676357298398 ; 0.005998109421480 ; 0.000000000000000 ];

%-- Image size:
nx = 1920;
ny = 1080;


%-- Various other variables (may be ignored if you do not use the Matlab Calibration Toolbox):
%-- Those variables are used to control which intrinsic parameters should be optimized

n_ima = 7;						% Number of calibration images
est_fc = [ 1 ; 1 ];					% Estimation indicator of the two focal variables
est_aspect_ratio = 1;				% Estimation indicator of the aspect ratio fc(2)/fc(1)
center_optim = 1;					% Estimation indicator of the principal point
est_alpha = 0;						% Estimation indicator of the skew coefficient
est_dist = [ 1 ; 1 ; 1 ; 1 ; 0 ];	% Estimation indicator of the distortion coefficients


%-- Extrinsic parameters:
%-- The rotation (omc_kk) and the translation (Tc_kk) vectors for every calibration image and their uncertainties

%-- Image #1:
omc_1 = [ 1.983963e+00 ; 2.244625e+00 ; -9.448716e-02 ];
Tc_1  = [ 2.729764e+00 ; -3.648266e+01 ; 1.189683e+02 ];
omc_error_1 = [ 2.898543e-02 ; 3.294006e-02 ; 5.879633e-02 ];
Tc_error_1  = [ 4.642020e+00 ; 7.244966e+00 ; 1.839435e+01 ];

%-- Image #2:
omc_2 = [ -2.070039e+00 ; -2.212749e+00 ; 2.965723e-01 ];
Tc_2  = [ 8.134444e+01 ; -3.754362e+01 ; 1.126389e+02 ];
omc_error_2 = [ 5.079182e-02 ; 5.495498e-02 ; 6.776371e-02 ];
Tc_error_2  = [ 4.772058e+00 ; 7.435037e+00 ; 1.929744e+01 ];

%-- Image #3:
omc_3 = [ 1.969233e+00 ; 2.211980e+00 ; 5.404511e-02 ];
Tc_3  = [ -5.110821e+01 ; -3.721133e+01 ; 1.166559e+02 ];
omc_error_3 = [ 2.817660e-02 ; 4.305254e-02 ; 7.443536e-02 ];
Tc_error_3  = [ 4.823530e+00 ; 7.347814e+00 ; 1.805496e+01 ];

%-- Image #4:
omc_4 = [ 1.969625e+00 ; 2.175465e+00 ; 1.288218e-01 ];
Tc_4  = [ -5.225445e+01 ; -1.762730e+01 ; 1.087125e+02 ];
omc_error_4 = [ 4.185091e-02 ; 5.955099e-02 ; 7.265841e-02 ];
Tc_error_4  = [ 4.494951e+00 ; 6.911697e+00 ; 1.676157e+01 ];

%-- Image #5:
omc_5 = [ 2.055685e+00 ; 2.309629e+00 ; 1.953818e-04 ];
Tc_5  = [ 5.605535e+00 ; -1.790445e+01 ; 1.033584e+02 ];
omc_error_5 = [ 2.862573e-02 ; 3.153465e-02 ; 5.886899e-02 ];
Tc_error_5  = [ 3.983437e+00 ; 6.378120e+00 ; 1.579389e+01 ];

%-- Image #6:
omc_6 = [ -1.967575e+00 ; -2.038981e+00 ; 5.583677e-01 ];
Tc_6  = [ 7.011752e+01 ; -1.984346e+01 ; 1.099208e+02 ];
omc_error_6 = [ 5.604859e-02 ; 6.904277e-02 ; 1.175781e-01 ];
Tc_error_6  = [ 4.621166e+00 ; 7.151127e+00 ; 1.784200e+01 ];

%-- Image #7:
omc_7 = [ 2.115864e+00 ; 2.204646e+00 ; -1.313860e-01 ];
Tc_7  = [ -6.236380e+00 ; -2.931430e+01 ; 6.746829e+01 ];
omc_error_7 = [ 2.461392e-02 ; 3.190929e-02 ; 6.494544e-02 ];
Tc_error_7  = [ 2.663480e+00 ; 4.074115e+00 ; 1.056924e+01 ];

