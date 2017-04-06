function [Xdx, XLd] = d_stereo_triangulation(xLx,xLy,xRx,xRy,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right)
%% complete differential, varables are xLx, xLy, xRx, xRy, in 1 pixel
% numeracal approximation
    [XL0,XR0] = stereo_triangulation([xLx,xLy]',[xRx,xRy]',om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
    [XL1,XR1] = stereo_triangulation([xLx+1,xLy]',[xRx,xRy]',om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
    [XL2,XR2] = stereo_triangulation([xLx,xLy+1]',[xRx,xRy]',om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
    [XL3,XR3] = stereo_triangulation([xLx,xLy]',[xRx+1,xRy]',om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
    [XL4,XR4] = stereo_triangulation([xLx,xLy]',[xRx,xRy+1]',om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
	% Xdx, first output is the mean of left and right camera
    XLd2 = (XL0-XL1).^2 + (XL0-XL2).^2 + (XL0-XL3).^2 + (XL0-XL4).^2 ;
    XRd2 = (XR0-XR1).^2 + (XR0-XR2).^2 + (XR0-XR3).^2 + (XR0-XR4).^2 ;
    XLd = sqrt(XLd2); XRd = sqrt(XRd2);
    Xdx = (XLd(1)+XRd(1))/2;
end