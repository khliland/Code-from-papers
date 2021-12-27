function [U,S,C,P] = PCP(Yh, X)
% Principal components of predictions

[U,S,V] = svd(Yh-mean(Yh),'econ');
C = V*S; % Y loadings
if nargin == 2
    X = X-mean(X);
    P = X'*U; % X loadings
else
    P = [];
end
