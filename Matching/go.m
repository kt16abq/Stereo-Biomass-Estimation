%% 
% Manually got the pixel position from R & L pictures
% by Frost Xu
% @ KAUST, VCC
%

SR = imread('C:\Users\Frost\Desktop\VCC\EdgeDetection\SR.bmp');
SL_P = imread('C:\Users\Frost\Desktop\VCC\EdgeDetection\SL_P.bmp');
SL = imread('C:\Users\Frost\Desktop\VCC\EdgeDetection\SL.bmp');
imPart = rgb2gray(SL_P);
imWhole = rgb2gray(SR);
imWholeL = rgb2gray(SL); 
scale = 10;
Lt = 100;
Lb = 2*Lt;
% BW1
I = imresize(imWholeL,[size(imWhole)]/scale);
optfun = @(x) (sum(sum(edge(I,'Canny',x)))-425)^2;
x = fminbnd(optfun,0,1);
SE = strel('disk', 13);
BW1 = edge(I,'Canny',x);
BW1 = imclose(BW1,SE);
imshow(BW1);
sum(sum(BW1));

% BW2
I = imresize(imWhole,[size(imWhole)]/scale);
optfun = @(x) (sum(sum(edge(I,'Canny',x)))-425)^2;
x = fminbnd(optfun,0,1);
BW2 = edge(I,'Canny',x);
BW2 = imclose(BW2,SE);
imshow(BW2);
sum(sum(BW2))

%% find similar
%% method 1: traversal
% [a,b] = size(BW1);
% x = zeros(a,b);
% for k=1:a
%     disp(k/a)
%     for l=1:b
%         x(k,l) = distance(BW1,BW2,k-a/2,l-b/2);
%     end
% end
%% method 2: fmincon
optfun = @(x) distance(BW1,BW2,x(1),x(2));
shift = fmincon(optfun,[0,0],[],[],[],[],[-size(BW1)]/2,[size(BW1)/2]);
shift = scale*shift;
%% check optfun
% rang = [-15:0.10:0];
% [k,l] = meshgrid(rang,rang);
% z = zeros(length(rang));
% for kk=1:length(rang)
%     disp(kk/length(rang))
%     for ll=1:length(rang);
%         z(kk,ll)=optfun([rang(kk),rang(ll)]);
%     end
% end

figure(1); hold on
subplot(221)
imshow(SL);view(2);
ht1 = title('Left (Click on this one)');
h1=rectangle('position',[0,0,100,100],'edgecolor','r');
subplot(223)
imshow(SR);view(2);
ht2 = title('Right (test)');
h2=rectangle('position',[0,0,100,100],'edgecolor','r');

flag = 1;
while flag
    waitforbuttonpress;
    point_temp = get(gca, 'currentpoint');
    x = round(point_temp(1,1)); plx = x;
    y = round(point_temp(1,2)); ply = y;
    
    subplot(221)
    h1.Position=[x-50,y-50,100,100];

    subplot(222)
    try
        imPartL = SL(y-50:y+50,x-50:x+50,:);
        imshow(imPartL);
        offsetL = [max(y-Lt,1),max(1,x-Lt)];
        imPartL = SL(offsetL(1):y+Lt,offsetL(2):x+Lt,:);
    catch
        disp('error')
    end
    rectangle('position',[-50,-50,50,50],'edgecolor','r');
    % line([-100,100],[0,0]);line([0,0],[-100,100]);
    
    subplot(223)
    x = x+shift(2);
    y = y+shift(1);
    h2.Position=[x-50,y-50,100,100];
    subplot(224)
    try
        imshow(SR(y-50:y+50,x-50:x+50,:));
    catch
        disp('error')
    end
    rectangle('position',[-50,-50,50,50],'edgecolor','r');
    res = input('OK? 1/0[1]: ');
        if isempty(res)
            res = 1;
        end
    flag = ~res;   
end

%% bottom
set(ht1,'String','Left (test)');
set(ht2,'String','Right (Click on this one)');

flag = 1;
while flag
    waitforbuttonpress;
    point_temp = get(gca, 'currentpoint');
    x = round(point_temp(1,1)); prx = x;
    y = round(point_temp(1,2)); pry = y;
    
    
    subplot(223)
    h2.Position=[x-50,y-50,100,100];
    subplot(224)
    try
        imPartR = SR(y-50:y+50,x-50:x+50,:);
        imshow(imPartR);
        offsetR = [max(y-Lb,1),max(1,x-Lb)];
        imPartR = SR(offsetR(1):y+Lb,offsetR(2):x+Lb,:);

    catch
        disp('error')
    end
    rectangle('position',[-50,-50,50,50],'edgecolor','r');
    res = input('OK? 1/0[1]: ');
        if isempty(res)
            res = 1;
        end
    flag = ~res;   
end

disp('finished')

disp('plx,ply,prx,pry:')

disp([plx,ply,prx,pry])


imPartL = rgb2gray(imPartL);
imPartR = rgb2gray(imPartR);
template = imPartL;
background = imPartR;


%% calculate padding
bx = size(background, 2); 
by = size(background, 1);
tx = size(template, 2); % used for bbox placement
ty = size(template, 1);

%% fft
% c = real(ifft2(fft2(background) .* fft2(template, by, bx)));
c = Myconv(template,background);
%% find peak correlation
[max_c, imax]   = max(abs(c(:)));
[ypeak, xpeak] = find(c == max(c(:)));
figure(2); surf(c), shading flat; % plot correlation 

%% display best match
hFig = figure(3);
hAx  = axes;
position = [xpeak(1), ypeak(1), tx, ty];
imshow(background, 'Parent', hAx);
imrect(hAx, position);
% imshow(template, 'Parent', hAx);
shift = [xpeak(1)-(bx-tx+1)/2,ypeak(1)-(by-ty+1)/2]
disp(shift)
    
figure(1)
subplot(224)
pry = pry ;%+ xpeak(1)-(bx-tx+1)/2;
prx = prx ;%+ ypeak(1)-(by-ty+1)/2;
    try
        imPartR = SR(pry-50:pry+50,prx-50:prx+50,:);
        imshow(imPartR);
        offsetR = [max(y-Lb,1),max(1,x-Lb)];
        imPartR = SR(offsetR(1):y+Lb,offsetR(2):x+Lb,:);
    catch
        disp('error')
    end

