function [ N ] = Medfilt2( M,l )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[a,b] = size(M);
l = 5;
Mp = [M,M,M;M,M,M;M,M,M];
N = medfilt2(Mp,[l,l]);
N = N(a+1:2*a,b+1:2*b);
end

