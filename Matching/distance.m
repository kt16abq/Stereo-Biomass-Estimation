function res = distance(BW1,BW2,x,y)
    % expand
    % x = round(x); y = round(y);
    % find neighber
    xo = x; yo = y;
    xl = floor(x); xu = xl+1;
    yl = floor(y); yu = yl+1;
    listxy = [xl,yl;xu,yl;xl,yu;xu,yu];
    [a,b] = size(BW1);
    BW1 = [zeros(size(BW1)),zeros(size(BW1)),zeros(size(BW1));zeros(size(BW1)),BW1,zeros(size(BW1));zeros(size(BW1)),zeros(size(BW1)),zeros(size(BW1))];
    BW2 = [zeros(size(BW2)),zeros(size(BW2)),zeros(size(BW2));zeros(size(BW2)),BW2,zeros(size(BW2));zeros(size(BW2)),zeros(size(BW2)),zeros(size(BW2))];
    res = 0;
    for k = 1:4
        x = listxy(k,1); y = listxy(k,2);
        res = res + (1-abs(x-xo))*(1-abs(y-yo))*sum(sum(BW1(a+1:2*a,b+1:2*b)==BW2(x+a+1:x+2*a,y+b+1:y+2*b)));
    end
    res = -res;
end