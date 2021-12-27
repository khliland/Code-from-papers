function [W,T,P,beta,Q] = pkcpls(X,y,A,yadd,cca)
% -----------------------------------------------------
% ------------------ KH Liland 2019 -------------------
% -----------------------------------------------------
% --------- Solution of the (C)PLS(2)-problem ---------
[nobj, nresp] = size(y);
X = bsxfun(@minus, X, mean(X)); 
y = bsxfun(@minus, y, mean(y));
if nargin < 4, yadd = []; 
else, yadd = bsxfun(@minus,yadd,mean(yadd)); end
if nargin < 5, cca = false; end

T = zeros(nobj,A); Q = zeros(A, nresp); Ry = T;
C = X*X';
if cca
    P = zeros(size(X,2), A); W = P;
end
for a = 1:A
    % ---------- Calculate scores ----------
    if cca
        tt = C*[y yadd];
        [~,w] = ccXY(tt,y,[]);       % CPLS
    else
        tt = C*y;
        [~,~,w] = svd(y'*tt,'econ'); % PLS2
    end
    t = tt*w(:,1);
    if a > 1
        t = t - T(:,1:a-1)*(T(:,1:a-1)'*t);
    end
    t = t/norm(t); T(:,a) = t;
    % ------------------ Deflate y --------------------
    Ry(:,a) = [y yadd]*w(:,1);
    Q(a,:) = t'*y; y = y - t*Q(a,:); 
    if cca
        Ctt = (C*t)*t';
        C   = C - Ctt - Ctt' + t*(t'*Ctt);
%         W(:,a) = 
    end
end
% -------- Calculate regression coefficients ----------
W = X'*Ry;
norm_W = vecnorm(W);
W = bsxfun(@rdivide, W, norm_W);
if nargout > 2
    P = X'*T;
end
if nargout > 3
    beta = bsxfun(@times, cumsum((W/triu(T' ...
        *bsxfun(@rdivide,C*Ry,norm_W))),2), ...
        reshape(Q,[1,A,nresp]));
end

%% Canonical correlations 
% (adaptation of MATLAB's canoncorr)
function [r,A] = ccXY(X,Y,wt)
% Computes the coefficients in canonical variates 
% between collumns of X and Y
[n,p1] = size(X); p2 = size(Y,2);

% Weighting of observations with regards to wt 
% (asumes weighted centering already performed)
if ~isempty(wt)
    X = rscale(X,wt);
    Y = rscale(Y,wt);
end

% Factoring of data by QR decomposition and elli-
% mination of internal linear dependencies in X and Y
[Q1,T11,perm1] = qr(X,0);       [Q2,T22,~] = qr(Y,0);
rankX = sum(abs(diag(T11)) > eps(abs(T11(1)))*max(n,p1));
rankY = sum(abs(diag(T22)) > eps(abs(T22(1)))*max(n,p2));
if rankX < p1
    Q1 = Q1(:,1:rankX); T11 = T11(1:rankX,1:rankX);
end
if rankY < p2
    Q2 = Q2(:,1:rankY);
end

% Economical computation of canonical coefficients 
% and canonical correlations
d = min(rankX,rankY);
if nargout == 1
    D = svd(Q1' * Q2,0);
    r = min(max(D(1:d), 0), 1);
else
    [L,D] = svd(Q1' * Q2,0);
    A     = T11 \ L(:,1:d) * sqrt(n-1);
    % Transform coeffs. to full size and correct order
    A(perm1,:) = [A; zeros(p1-rankX,d)];
    r = min(max(diag(D(1:d)), 0), 1); % Canonical corr.
end
