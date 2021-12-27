function [W,P,T,beta,Q,Xsnitt,Ysnitt] = pls(pc,X,Y,wt)
% Ordinary PLS (non-orthogonal scores)

if nargin <4
    wt = ones(size(X,1),1);
end

[X, Xsnitt, n, p] = Center(X,wt);
[y, Ysnitt]       = Center(Y,wt);

Q = zeros(pc,1);
T = zeros(n,pc);
W = zeros(p,pc);
P = zeros(p,pc);
beta = zeros(p,pc);
for a= 1:pc
    c = 1./sqrt((y'*X)*X'*y);
    W(:,a) = c*X'*y;
    T(:,a) = X*W(:,a);
    P(:,a) = (X'*T(:,a))/(T(:,a)'*T(:,a));
    Q(a) = (y'*T(:,a))/(T(:,a)'*T(:,a));
    X = X - T(:,a)*P(:,a)';
    y = y - T(:,a)*Q(a);
    if nargout > 3
        beta(:,a) =  W(:,1:a)/(P(:,1:a)'*W(:,1:a))*Q(1:a);
    end
end

%% Weighted centering
function [X, mX, n, p] = Center(X,wt)
% Centering of the data matrix X by subtracting the weighted column means
% according to the nonegative weights wt
[n,p] = size(X);

% Calculation of column means:
if nargin == 2 && ~isempty(wt)
    mX = (wt'*X)./sum(wt);
else
    mX = mean(X);
end

% Centering of X, similar to: %X = X-ones(n,1)*mX;
X = X-repmat(mX,n,1);
