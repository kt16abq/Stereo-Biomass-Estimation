 xmin = kron(1:7,ones(7-2+1,1)); 
xmax = bsxfun(@plus, 1:7,[ 2:7]');
ymin = xmin; ymax = xmax;
a = {}; b = a;
for k = 1:length(xmin(:))
   a{k} = xmin(k):xmax(k);
   b{k} = ymin(k):ymax(k);   
end
[ai,bi] = meshgrid(1:length(a),1:length(a));

for N = 1:length(ai(:))
   [A,B] = meshgrid(a{ai(N)},b{bi(N)});
   S(N) = {[A(:),B(:)]};
end