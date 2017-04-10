%% 
% Manually got the pixel position from R & L pictures
% by Frost Xu
% @ KAUST, VCC
%

%% Load toolbox, pictures and parameters
clc; clear; close all;
addpath('TOOLBOX_calib')
% it is the default name from toolbox
load('Calib_Results_stereo.mat')
SR = imread('SR.bmp');
SL = imread('SL.bmp');

%% manually click, and then optimize result
% show two picture
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
HTLR = zeros(2,2,2); % save pixel result to HTLR, 2x2x2 is 4 points in 2 dimension. % Head and Tail % Left and Right
% useful strings
strHT = {'head','tail'};
strLR = {'left','right','L','R'};
for k=1:2 % Head and Tail
    for l=1:2 % Left and Right
        for flag = 1:5
            subplot(2,2,l); axis off
            msg = ['Pls click its ', strHT{k}, ' in ', strLR{l}, ' vision,  ', num2str(5-flag),' times more.'];
            ht = text(100,100,msg);
            % user click
            waitforbuttonpress;
            % resize the other figure
            subplot(2,2,3-l) 
            zoom(1/1E3)
            subplot(2,2,l)
            delete(ht)
            % read pixel position, (x,y)
            point_temp = get(gca, 'currentpoint');
            x = round(point_temp(1,1)); plx = x;
            y = round(point_temp(1,2)); ply = y;
            HTLR(:,k,l)=[x,y];
            subplot(2,2,l)
            eval(['h',num2str(l),'.Position=[x-50,y-50,100,100];'])
            h1.Position=[x-50,y-50,100,100];
            subplot(2,2,l+2)
            axis off; axis equal
            % try to save local ragion
            try
                eval(['S = S',strLR{l+2},';']); %S = SL;            
                eval(['imPart',strLR{l+2},'=S(y-50:y+50,x-50:x+50,:);']);          
                imPart = S(y-50:y+50,x-50:x+50,:);
                surf(rgb2gray(imPart));view(2);shading interp
            catch
                disp('error')
            end
            line([0,100],[50,50]);line([50,50],[0,100]); % reference line, failed
        end
    % every two pictures, refine the pixel pair location by
    if(l==2) 
        [minx,miny] = Refinement(imPartL,imPartR);
        HTLR(:,k,2)=HTLR(:,k,2)+ [minx;miny]/2;
        HTLR(:,k,1)=HTLR(:,k,1)- [minx;miny]/2;
    end
    end
end
close all

%% Triangulation
% replace it when calibrated
[XLH,XRH] = stereo_triangulation(HTLR(:,1,1),HTLR(:,1,2),om,T,fc_left,cc_left,kc_left,0,fc_right,cc_right,kc_right,0);
[XLT,XRT] = stereo_triangulation(HTLR(:,2,1),HTLR(:,2,2),om,T,fc_left,cc_left,kc_left,0,fc_right,cc_right,kc_right,0);
lengthL = XLH-XLT;  lengthL = sqrt(lengthL'*lengthL);
lengthR = XRH - XRT; lengthR = sqrt(lengthR'*lengthR);
lengthShark = (lengthL+lengthR)/2;
disp('Length of Shark is:')
disp(lengthShark)
