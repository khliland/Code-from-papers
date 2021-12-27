function [T,P,beta] = pcr(X,y,k)
% ----------------------------------------------------------
% ------------------- KH Liland 2019 -----------------------
% ----------------------------------------------------------
% ------------- Solution of the PCA/R-problems -------------
X = bsxfun(@minus, X, mean(X)); 
[U,S,P] = svd(X,'econ');
s = diag(S(1:k,1:k));
T = U(:,1:k).*s';
P = P(:,1:k);
if nargout > 2
    y = y-mean(y);
    beta = P*(s.^2.\(T'*y));
%     beta = P*((T'*T)\(T'*y));
end
