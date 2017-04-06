function [x_kk] = normalize_pixel_1(xn,fc,cc,kc,alpha_c)
%% The revarse function of normalize_pixel
%normalize
%
%[xn] = normalize_pixel(x_kk,fc,cc,kc,alpha_c)
%
%Computes the normalized coordinates xn given the pixel coordinates x_kk
%and the intrinsic camera parameters fc, cc and kc.
%
%INPUT: x_kk: Feature locations on the images
%       fc: Camera focal length
%       cc: Principal point coordinates
%       kc: Distortion coefficients
%       alpha_c: Skew coefficient
%
%OUTPUT: xn: Normalized feature locations on the image plane (a 2XN matrix)
%
%Important functions called within that program:
%
%comp_distortion_oulu: undistort pixel coordinates.
% to be optimaized

if nargin < 5,
   alpha_c = 0;
   if nargin < 4;
      kc = [0;0;0;0;0];
      if nargin < 3;
         cc = [0;0];
         if nargin < 2,
            fc = [1;1];
         end;
      end;
   end;
end;

lb = [0;0];  ub =cc*2;
x_kk0 = (lb+ub) /2;
% use fmincon to find input
% you can optimaze this function
optfunn = @(x)  sum((normalize_pixel(x,fc,cc,kc,alpha_c)-xn).^2);
options = optimoptions('fmincon');
options = optimoptions(options,'Display', 'off');

[x_kk,fval,exitflag,output] = ...
fmincon(optfunn,x_kk0,[],[],[],[],lb,ub,[],options);

