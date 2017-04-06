function res= Myconv( part, whole )
%MYCONV Summary of this function goes here
%   Detailed explanation goes here
% 
% sigma = 1.6;
% gausFilter = fspecial('average',[5 5]);
part = double(part);part=Medfilt2(part,3);
% part=imfilter(part,gausFilter,'replicate');
whole = double(whole);whole = Medfilt2(whole,3);
% whole=imfilter(whole,gausFilter,'replicate');

[a,b] = size(part);
res = zeros(size(whole,1)-a+1, size(whole,2)-b+1);
p2 =(part(:)'*part(:));
for k=1:size(res,1)
    for l = 1:size(res,2)
        partW = whole(k:k+a-1,l:l+b-1);
        res(k,l) = (part(:)'*partW(:))/(p2+partW(:)'*partW(:));
        
    end
end

end

