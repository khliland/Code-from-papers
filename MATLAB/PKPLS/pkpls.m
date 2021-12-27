function [W,T,P,beta,q] = pkpls(X,y,k)
% -----------------------------------------------------
% ----------------- KH Liland 2019 --------------------
% -----------------------------------------------------
% ---------- Solution of the PLS1-problem -------------
X = bsxfun(@minus, X, mean(X)); y = y-mean(y);
T = zeros(size(X,1),k); q = T(1,:); Ry = T;
C = X*X';
for i = 1:k
    t = C*y; 
    if i > 1
        t = t - T(:,1:i-1)*(T(:,1:i-1)'*t);
    end
    t = t/norm(t); T(:,i) = t;
    % ------------------- Deflate y -------------------
    Ry(:,i) = y;
    q(i) = y'*t; y = y - q(i)*t; 
end
% ------- Calculate regression coefficients -----------
W = X'*Ry;
norm_W = sqrt(sum(W.^2));
W = bsxfun(@rdivide, W, norm_W);
if nargout > 2
    P = X'*T;
end
if nargout > 3
    beta = cumsum(bsxfun(@times,W/triu(P'*W), q),2);
end
