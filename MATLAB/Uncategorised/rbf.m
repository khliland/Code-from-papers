function [K, sqeuc] = rbf(X, lambda, sqeuc)
%% Radial basis function of X
% This function either computes RBFs from skratch or from a
% precomputed (reused) squared Euclidean distance matrix
if ~isempty(X)
    XX = X*X'; 
    dx = diag(XX);
    sqeuc = dx+dx' - 2.*XX;
end
K = exp(-lambda.*sqeuc);
