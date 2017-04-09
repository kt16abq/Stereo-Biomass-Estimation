%% 
% Manually got the pixel position from R & L pictures
% by Frost Xu
% @ KAUST, VCC
%
clc; clear; close all;
addpath('C:\Users\xum\Downloads\toolbox_calib\TOOLBOX_calib')
load('inPara.mat')
SR = imread('SR.bmp');
SL = imread('SL.bmp');

figure(1); hold on; axis off
subplot(111)
ht = text(0,0.5,{'Manually got the pixel position from R & L pictures [click to continue]'});
waitforbuttonpress;
subplot(221)
imshow(SL);view(2);
ht1 = title('Left Picture');
h1=rectangle('position',[0,0,100,100],'edgecolor','r');
subplot(222)
imshow(SR);view(2);
ht2 = title('Right Picture');
h2=rectangle('position',[0,0,100,100],'edgecolor','r');
HTLR = zeros(2,2,2);
strHT = {'head','tail'};
strLR = {'left','right','L','R'};
for k=1:2
    for l=1:2
        for flag = 1:5
            subplot(2,2,l); axis off
            msg = ['Pls click its ', strHT{k}, ' in ', strLR{l}, ' vision,  ', num2str(5-flag),' times more.'];
            ht = text(100,100,msg);
           waitforbuttonpress;
            delete(ht)
            point_temp = get(gca, 'currentpoint');
            x = round(point_temp(1,1)); plx = x;
            y = round(point_temp(1,2)); ply = y;
            HTLR(:,k,l)=[x,y];
            subplot(2,2,l)
            eval(['h',num2str(l),'.Position=[x-50,y-50,100,100];'])
            h1.Position=[x-50,y-50,100,100];
            subplot(2,2,l+2)
            axis off; axis equal
            try
                eval(['S = S',strLR{l+2},';']); %S = SL;                
                imPart = S(y-50:y+50,x-50:x+50,:);
                surf(rgb2gray(imPart));view(2);shading interp
            catch
                disp('error')
            end
            line([0,100],[50,50]);line([50,50],[0,100]);
        end

    end
end
close all

%% Triangulation
% replace it when calibrated
om=[ 0.05106   -0.14839  0.05472 ]'; T = [ -412.58240   -19.01564  -48.22109 ]' ;
[XLH,XRH] = stereo_triangulation(HTLR(:,1,1),HTLR(:,1,2),om,T,fc_left,cc_left,kc_left,0,fc_right,cc_right,kc_right,0);
[XLT,XRT] = stereo_triangulation(HTLR(:,2,1),HTLR(:,2,2),om,T,fc_left,cc_left,kc_left,0,fc_right,cc_right,kc_right,0);
lengthL = XLH-XLT;  lengthL = sqrt(lengthL'*lengthL);
lengthR = XRH - XRT; lengthR = sqrt(lengthR'*lengthR);
lengthShark = (lengthL+lengthR)/2;
disp('Length of Shark is:')
disp(lengthShark)
