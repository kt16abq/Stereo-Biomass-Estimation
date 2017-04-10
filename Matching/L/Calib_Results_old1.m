% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly executed under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 1157.136142679897800 ; 1235.007115814959200 ];

%-- Principal point:
cc = [ 1003.407137512057900 ; 548.887019389192800 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ -0.241434955516505 ; 0.061130503324608 ; 0.004805388220738 ; -0.002193505668349 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 177.075022105001640 ; 186.951953714531670 ];

%-- Principal point uncertainty:
cc_error = [ 47.562490756788208 ; 78.322532415180959 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.060933488753905 ; 0.034621695906151 ; 0.007658901520968 ; 0.006635622147588 ; 0.000000000000000 ];

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
omc_1 = [ 1.984949e+00 ; 2.246451e+00 ; -9.229121e-02 ];
Tc_1  = [ 3.140390e+00 ; -3.621862e+01 ; 1.185463e+02 ];
omc_error_1 = [ 2.966766e-02 ; 3.403788e-02 ; 6.137671e-02 ];
Tc_error_1  = [ 4.947222e+00 ; 7.401703e+00 ; 1.837875e+01 ];

%-- Image #2:
omc_2 = [ -2.065089e+00 ; -2.207453e+00 ; 3.005101e-01 ];
Tc_2  = [ 8.166109e+01 ; -3.724756e+01 ; 1.118919e+02 ];
omc_error_2 = [ 5.301701e-02 ; 5.749385e-02 ; 7.076155e-02 ];
Tc_error_2  = [ 5.035470e+00 ; 7.598760e+00 ; 1.932099e+01 ];

%-- Image #3:
omc_3 = [ NaN ; NaN ; NaN ];
Tc_3  = [ NaN ; NaN ; NaN ];
omc_error_3 = [ NaN ; NaN ; NaN ];
Tc_error_3  = [ NaN ; NaN ; NaN ];

%-- Image #4:
omc_4 = [ 1.965548e+00 ; 2.169320e+00 ; 1.313631e-01 ];
Tc_4  = [ -5.180343e+01 ; -1.733554e+01 ; 1.080537e+02 ];
omc_error_4 = [ 4.533658e-02 ; 6.561399e-02 ; 7.697518e-02 ];
Tc_error_4  = [ 4.798122e+00 ; 7.063028e+00 ; 1.670973e+01 ];

%-- Image #5:
omc_5 = [ 2.055838e+00 ; 2.309341e+00 ; 6.498054e-03 ];
Tc_5  = [ 5.955979e+00 ; -1.765175e+01 ; 1.029292e+02 ];
omc_error_5 = [ 2.990873e-02 ; 3.259224e-02 ; 6.031734e-02 ];
Tc_error_5  = [ 4.241395e+00 ; 6.514147e+00 ; 1.576139e+01 ];

%-- Image #6:
omc_6 = [ -1.962625e+00 ; -2.033738e+00 ; 5.642343e-01 ];
Tc_6  = [ 7.043390e+01 ; -1.955280e+01 ; 1.092816e+02 ];
omc_error_6 = [ 5.799612e-02 ; 7.114029e-02 ; 1.196368e-01 ];
Tc_error_6  = [ 4.866237e+00 ; 7.307465e+00 ; 1.784716e+01 ];

%-- Image #7:
omc_7 = [ 2.117566e+00 ; 2.207079e+00 ; -1.279855e-01 ];
Tc_7  = [ -6.004047e+00 ; -2.917627e+01 ; 6.721457e+01 ];
omc_error_7 = [ 2.538029e-02 ; 3.312526e-02 ; 6.816500e-02 ];
Tc_error_7  = [ 2.837407e+00 ; 4.163869e+00 ; 1.056669e+01 ];

