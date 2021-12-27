function [T,P,beta] = pkpcr(X,y,k)
% -----------------------------------------------------
% ----------------- KH Liland 2019 --------------------
% -----------------------------------------------------
% ---------- Solution of the PCA/R-problems -----------
X = bsxfun(@minus, X, mean(X)); 
C = X*X';
[U,S] = eig(C,'vector');
T = U(:,end:-1:end-k+1).*sqrt(S(end:-1:end-k+1,1))';
% Alternatively using SVD
% [U,S] = svd(C,'econ'); 
% T = U(:,1:k).*sqrt(diag(S(1:k,1:k)))';
if nargout > 1
    P = (T\X)';
end
if nargout > 2
    y = y-mean(y);
    beta = cumsum(P.*(S(end:-1:end-k+1,1).\(T'*y))',2);
    % Alternatively using SVD
    % beta = cumsum(P.*(diag(S(1:k,1:k)).\(T'*y))',2);
end
