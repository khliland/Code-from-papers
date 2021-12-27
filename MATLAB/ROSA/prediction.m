function [ypred, rmse] = prediction(X,Y,beta,Xnew,Ynew)
% [ypred,rmse] = prediction(X,Y,beta,Xnew,Ynew)
% ypred = (Xnew - mean(X)) * beta + mean(Y)

if nargin < 4
    Xnew = X;
    Ynew = Y;
end

ypred = bsxfun(@plus,bsxfun(@minus,Xnew,mean(X))*beta,mean(Y));
if nargin > 4
    rmse  = sqrt(mean((Ynew-ypred).^2));
end