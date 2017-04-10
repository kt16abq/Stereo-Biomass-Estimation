function [minx, miny] = Refinement(impartL,impartR)
%% quadratic polynomial interpolation, *2-D*
% improved from
% Stereo Matching with Color-Weighted Correlation,
% Hierarchical Belief Propagation and Occlusion
% Handling, Qingxiong Yang

[a,b] = size(impartL);
ca = round(a/2); cb = round(b/2);
marg = round(ca/3); a2 = min(ca - marg,marg) -1;
costM = zeros(marg*2+1);
%% cost matrix
for k = -marg:marg
    for l = -marg:marg
        costM(marg+k+1,marg+l+1) = ...
            Cost(impartL(ca-a2:ca+a2,ca-a2:ca+a2,:),impartR(ca-a2+k:ca+a2+k,ca-a2+l:ca+a2+l,:));
    end
end
[dx,dy] = find(costM == min(min(costM)));
%% edge?
% remember to matrix2image index
if dx==1 || dx==marg*2+1 || dy==1 || dy==marg*2+1
    disp('warning, in-accurate click pair.')
    miny = dx; minx = dy; % matrix2image
else
    disp('good job, optimize result..')
    
    cx0 = Cost(impartL(ca-a2:ca+a2,ca-a2:ca+a2,:),...
        impartR([ca-a2:ca+a2]+dx-marg-1,[ca-a2:ca+a2]+dy-marg-1,:));
    cxp = Cost(impartL(ca-a2:ca+a2,ca-a2:ca+a2,:),...
        impartR([ca-a2:ca+a2]+dx+1-marg-1,[ca-a2:ca+a2]+dy-marg-1,:));
    cxn = Cost(impartL(ca-a2:ca+a2,ca-a2:ca+a2,:),...
        impartR([ca-a2:ca+a2]+dx-1-marg-1,[ca-a2:ca+a2]+dy-marg-1,:));
    miny = dx - (cxp-cxn)/(cxp+cxn-2*cx0)/2;
    
    cy0 = Cost(impartL(ca-a2:ca+a2,ca-a2:ca+a2,:),...
        impartR([ca-a2:ca+a2]+dx-marg-1,[ca-a2:ca+a2]+dy-marg-1,:));
    cyp = Cost(impartL(ca-a2:ca+a2,ca-a2:ca+a2,:),...
        impartR([ca-a2:ca+a2]+dx-marg-1,[ca-a2:ca+a2]+dy+1-marg-1,:));
    cyn = Cost(impartL(ca-a2:ca+a2,ca-a2:ca+a2,:),...
        impartR([ca-a2:ca+a2]+dx-marg-1,[ca-a2:ca+a2]+dy-1-marg-1,:));
    minx = dy - (cyp-cyn)/(cyp+cyn-2*cy0)/2; % matrix2image
end
%% margin 
minx = minx -marg-1; miny = miny -marg-1;
end

function c = Cost( p1, p2 )
%% See paper
k = 0.5; p1 = double(p1); p2 = double(p2);
if length(size(p1))==3
    c = 0;
    for k = 1:size(p1,3)
        c = c + Cost( p1(:,:,k), p2(:,:,k) )/size(p1,3);
    end
else
    c = [cCos(p1(:),p2(:)),cAD(p1(:),p2(:))]*[k,1-k]';
end

end

function c = cCos( p1, p2 )
c = 1 - abs(2*p1'*p2 /(p1'*p1+p2'*p2));
end


function c = cAD( p1, p2 )
dp = p1 - p2;
c = 0.5* dp'*dp /(p1'*p1+p2'*p2);
end