function [xL,xR,XR,flag] = stereo_triangulation_1(X0,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right),
%% The revarse function of stereo_triangulation
% [XL,XR] = stereo_triangulation(xL,xR,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right),
%
% Function that computes the position of a set on N points given the left and right image projections.
% The cameras are assumed to be calibrated, intrinsically, and extrinsically.
%
% Input:
%           xL: 2xN matrix of pixel coordinates in the left image
%           xR: 2xN matrix of pixel coordinates in the right image
%           om,T: rotation vector and translation vector between right and left cameras (output of stereo calibration)
%           fc_left,cc_left,...: intrinsic parameters of the left camera  (output of stereo calibration)
%           fc_right,cc_right,...: intrinsic parameters of the right camera (output of stereo calibration)
%
% Output:
%
%           XL: 3xN matrix of coordinates of the points in the left camera reference frame
%           XR: 3xN matrix of coordinates of the points in the right camera reference frame
%
% Note: XR and XL are related to each other through the rigid motion equation: XR = R * XL + T, where R = rodrigues(om)
% For more information, visit http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/example5.html
%
%
% (c) Jean-Yves Bouguet - Intel Corporation - April 9th, 2003
% optimized by Frost Xu

%% baseline and relative angle 
B = sqrt(T'*T); tr = om(2);R = rodrigues(om);
% load('inPara.mat') cause too much time
th = 94.4/180*pi;
xL=[0;0];
xR=[0,0];
% transfer functions
XL = rodrigues(-om/2)*X0+rodrigues(-om/2)*(B*[1; 0; 0 ])/2;
XR = R*XL + T;
XR2 = rodrigues(+om/2)*X0+rodrigues(om/2)*(-B*[1; 0; 0 ])/2;
% disp(XR-XR2)
flag = 0;
xt = XL./  XL(3) ;
xtt = XR./XR(3);
% the point is in the view of feild?
if abs(xt(1))>tan(th/2)*0.8 || abs(xtt(1))>tan(th/2)*0.8
    flag = -1;
    return
end
% Extend the normalized projections in homogeneous coordinates
xt = xt(1:2);
xtt = xtt(1:2);
[xL] = normalize_pixel_1(xt,fc_left,cc_left,kc_left,alpha_c_left);
[xR] = normalize_pixel_1(xtt,fc_right,cc_right,kc_right,alpha_c_right);

