% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly executed under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 854.787816190537000 ; 921.644201414607780 ];

%-- Principal point:
cc = [ 888.187635391396160 ; 576.234078706345710 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ -0.209653764839616 ; 0.052457281974583 ; -0.002806981764025 ; -0.005088026414851 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 166.142675059921570 ; 180.157561607398410 ];

%-- Principal point uncertainty:
cc_error = [ 54.976686718007677 ; 72.365751658132154 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.065671954343804 ; 0.036970523560448 ; 0.007559495108379 ; 0.007944980649906 ; 0.000000000000000 ];

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
omc_1 = [ 2.011475e+00 ; 2.304001e+00 ; -2.096889e-01 ];
Tc_1  = [ -1.702858e+01 ; -4.111310e+01 ; 1.114336e+02 ];
omc_error_1 = [ 2.102733e-02 ; 4.419704e-02 ; 7.496415e-02 ];
Tc_error_1  = [ 7.290682e+00 ; 8.453016e+00 ; 2.162763e+01 ];

%-- Image #2:
omc_2 = [ -2.027529e+00 ; -2.213921e+00 ; 3.895424e-01 ];
Tc_2  = [ 6.140414e+01 ; -4.216388e+01 ; 1.010035e+02 ];
omc_error_2 = [ 4.965232e-02 ; 4.403165e-02 ; 5.940503e-02 ];
Tc_error_2  = [ 7.328716e+00 ; 8.239360e+00 ; 2.124806e+01 ];

%-- Image #3:
omc_3 = [ NaN ; NaN ; NaN ];
Tc_3  = [ NaN ; NaN ; NaN ];
omc_error_3 = [ NaN ; NaN ; NaN ];
Tc_error_3  = [ NaN ; NaN ; NaN ];

%-- Image #4:
omc_4 = [ 2.028799e+00 ; 2.247424e+00 ; 3.228773e-02 ];
Tc_4  = [ -7.281461e+01 ; -2.243617e+01 ; 1.070738e+02 ];
omc_error_4 = [ 3.370840e-02 ; 5.643046e-02 ; 8.499133e-02 ];
Tc_error_4  = [ 6.775813e+00 ; 8.896129e+00 ; 2.147623e+01 ];

%-- Image #5:
omc_5 = [ 2.058624e+00 ; 2.309749e+00 ; -1.507480e-01 ];
Tc_5  = [ -1.575882e+01 ; -2.204199e+01 ; 9.647542e+01 ];
omc_error_5 = [ 1.937152e-02 ; 3.396601e-02 ; 6.693512e-02 ];
Tc_error_5  = [ 6.219978e+00 ; 7.425303e+00 ; 1.868086e+01 ];

%-- Image #6:
omc_6 = [ -1.963293e+00 ; -2.063583e+00 ; 5.667570e-01 ];
Tc_6  = [ 4.991818e+01 ; -2.461384e+01 ; 9.910494e+01 ];
omc_error_6 = [ 4.327388e-02 ; 4.270863e-02 ; 1.081855e-01 ];
Tc_error_6  = [ 6.831896e+00 ; 8.013617e+00 ; 1.936770e+01 ];

%-- Image #7:
omc_7 = [ -2.149738e+00 ; -2.277073e+00 ; 2.346643e-01 ];
Tc_7  = [ -3.035904e+01 ; -3.227007e+01 ; 6.507480e+01 ];
omc_error_7 = [ 5.008062e-02 ; 1.455238e-02 ; 8.494313e-02 ];
Tc_error_7  = [ 4.253231e+00 ; 5.173375e+00 ; 1.287217e+01 ];

