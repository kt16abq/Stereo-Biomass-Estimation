%%
% Optimization for stereo structures
% by Frost, @ VCC
clear; clc; close all

%% add toolbox  and parameters
% to learn toolbox, see more information from http://www.vision.caltech.edu/bouguetj/calib_doc/
% inPara.mat include the instric parameters of cerrent device, two GoPros
% threshold, whose unit is mm/px, is the largest acceptable error
addpath('C:\Users\xum\Downloads\toolbox_calib\TOOLBOX_calib')
load('inPara.mat')
threshold = 30;

%% initial value, 
% x0(1), distance, is the suggest distance from sharks and people 
% x0(2) is relative angle, \theta_r, tr
x0 = [distance, 0]; %parameters is from current rig

%% two bonds
lb = [distance/5,-pi/4]; ub=[distance*2,pi/4];

%% Start with the default options
options = optimoptions('fmincon');
options = optimoptions(options,'Display', 'off');
options = optimoptions(options,'MaxFunctionEvaluations', 6000);
options = optimoptions(options,'PlotFcn', { @optimplotfval });
options = optimoptions(options,'Algorithm', 'active-set');
% local minima algorithm is commented
% [x,fval,exitflag,output,lambda,grad,hessian] = ...
%     fmincon(@(x)optfun(x),x0,[],[],[],[],lb,ub,@mycon,options);
problem = createOptimProblem('fmincon', ...
        'objective', @(x) optfun(x),   ...
        'x0',x0, 'lb',lb,'ub',ub,                                       ...
        'nonlcon', @(x) mycon(x),...
        'options',options);
 % use global search meathod to find global minima
ms = MultiStart;
% 50 50 local solvers
[x,Fx] = run(ms,problem,50);

%% error distribution
% IMPORTANT, before this part, a better way to get parameters is
% to re-calibrate cerrent stereo system
tr = x(2); % tr should become a real number
dist=x(1);
% y, vertical arix is sopposed to be zero
[X, Z] = meshgrid(...
    linspace(-distance*cos(th/2)-B,distance*cos(th/2)+B,200),linspace(distance/5,distance*1.5,200));
D = zeros(size(X));
% calculate error in every position
for k = 1:length(X(:))
    disp(k/length(X(:)))
    XL = [X(k); 0; Z(k)];
    [xL,xR,XR,flag] = stereo_triangulation_1(XL,[0; tr; 0 ],B*[-cos(tr/2); 0; sin(tr/2) ],fc_left,cc_left,kc_left,0,fc_right,cc_right,kc_right,0);
    % the point is in the view of field?
    if flag ==0
        Xdx = d_stereo_triangulation(xL(1),xL(2),xR(1),xR(2),[0; tr; 0 ],B*[-cos(tr/2); 0; sin(tr/2) ], fc_left,cc_left,kc_left,0,fc_right,cc_right,kc_right,0);
    else 
        Xdx = -threshold;
    end
    % larger than threshold?
    if Xdx>threshold
        D(k) = threshold; % fix it
    else
        D(k) = Xdx;
    end
end
Tr = linspace(tr-0.5,tr+0.5,500); Value = zeros(size(Tr));
for k=1:500
    Value(k) = optfun([x(1),Tr(k)]);
end
figure(3)
plot(Tr,Value)
% move down the whole distribution to get a better display
D = D - threshold;
for k = 1:length(X(:))
    if D(k)==0
        D(k) = -threshold*2;
    end
end

%% display
figure(2);hold on
surf(X,Z,D); shading interp; view(2)
title('Geometric Pattern & Error Distribution')

%Geometric Pattern
% define three colors
color1 = [127,201,127]/256;
color2 = [190,174,212]/256;
color3 =[253,192,134]/256;
ylabel('z (world coordinate, mm)')
xlabel('x (world coordinate, mm)')
length1 = distance/3/1.5; length2=distance*2;
% put two cameras on canvas
scatter(B/2,0,'filled','MarkerFaceColor',color2);scatter(-B/2,0,'filled','MarkerFaceColor',color1);
% centre lines and bounds for each cameras
line([-B/2,-B/2+length1*sin(tr/2)],[0,0+length1*cos(tr/2)],'color',color1);
line([B/2,B/2-length1*sin(tr/2)],[0,0+length1*cos(tr/2)],'color',color2);
line([-B/2,-B/2+length2*sin(tr/2+th/2)], [0,0+length2*cos(tr/2+th/2)],'color',color1);
line([B/2,B/2+length2*sin(-tr/2+th/2)],  [0,0+length2*cos(-tr/2+th/2)],'color',color2);
line([-B/2,-B/2+length2*sin(tr/2-th/2)],  [0,0+length2*cos(tr/2-th/2)],'color',color1);
line([B/2,B/2+length2*sin(-tr/2-th/2)],   [0,0+length2*cos(-tr/2-th/2)],'color',color2);
% sharks
line([-L/2,-L/2],[0,length2 ],'color',color3,'lineWidth',2);
line([L/2,L/2],[0,length2 ],'color',color3,'lineWidth',2);
line([-L/2,L/2],[dist,dist ],'color',color3,'lineWidth',2);
% colorbar
hcb=colorbar;
set(hcb,'Limits',[-2*threshold,0])
set(hcb,'YTick',[-2*threshold,-threshold,-eps(0)])
set(hcb,'TickLabels',{'invalid','zero error',[num2str(threshold) ' px/mm']})




%% Then several simple function
%% optfun, for optimization
% x is 2*1 array. First is the distance, next is the raletive angle in
% degree
function value = optfun(x) 
    tr = x(2);
    load('inPara.mat')
    [xL,xR,~,flag] = stereo_triangulation_1([-L/2;0;x(1)],[0; tr; 0 ],...
        B*[-cos(tr/2); 0; sin(tr/2) ],fc_left,cc_left,kc_left,0,fc_right,cc_right,kc_right,0);
    if flag ==0
        value = d_stereo_triangulation(xL(1),xL(2),xR(1),xR(2),[0; tr; 0 ],...
            B*[-cos(tr/2); 0; sin(tr/2) ], fc_left,cc_left,kc_left,0,fc_right,cc_right,kc_right,0);
    else 
        value = 9999;
    end
end
%% optcon, constrain of optimization
function [c,ceq] = mycon(x)
    c = optc(x);      % Compute nonlinear inequalities at x.
    ceq = optceq(x);  % Compute nonlinear equalities at x.
end 
% Compute nonlinear inequalities at x.
function neg = optc(x) 
    tr = x(2);
    load('inPara.mat')
    % head should in the view
    [xL,xR,~,flag] = stereo_triangulation_1([-L/2;0;x(1)],[0; tr; 0 ],...
        B*[-cos(tr/2); 0; sin(tr/2) ],fc_left,cc_left,kc_left,0,fc_right,cc_right,kc_right,0);    
    neg = [ -(2*flag+1)];
    % tail should in the view
    [xL,xR,~,flag] = stereo_triangulation_1([L/2;0;x(1)],[0; tr; 0 ],...
        B*[-cos(tr/2); 0; sin(tr/2) ],fc_left,cc_left,kc_left,0,fc_right,cc_right,kc_right,0);    
    neg = [neg; -(2*flag+1)];
end
% Compute nonlinear equalities at x.
function zer = optceq( x ) 
    zer = 0;
end