function [W,P,T,beta,Q,Xsnitt,Ysnitt, ssqx, ssqy] = pls2(pc,X,Y,wt)
% Ordinary PLS2 (non-orthogonal scores)

if nargin <4
    wt = [];
end

[X, Xsnitt, n, p]   = Center(X,wt);
[Y, Ysnitt, ~, py] = Center(Y,wt);
if nargout > 7
    ssqx = [sum(X(:).^2);zeros(pc,1)];   % Total variation in the data
    ssqy = [sum(Y(:).^2);zeros(pc,1)];  % Total variation in the data
end
Q = zeros(py,pc);
T = zeros(n,pc);
W = zeros(p,pc);
P = zeros(p,pc);
beta = zeros(p,py,pc);
for a=1:pc
    [~,~,W(:,a)] = svds(Y'*X,1);
    T(:,a) = X*W(:,a);
    Q(:,a) = (Y'*T(:,a))/(T(:,a)'*T(:,a));
    P(:,a) = (X'*T(:,a))/(T(:,a)'*T(:,a));
    X = X - T(:,a)*P(:,a)';
    Y = Y - T(:,a)*Q(:,a)';
    if nargout > 3
        beta(:,:,a) =  W(:,1:a)/(P(:,1:a)'*W(:,1:a))*Q(:,1:a)';
    end
    if nargout > 7 % Residual X-variance
        ssqx(a+1,1) = sum(X(:).^2); 
        ssqy(a+1,1) = sum(Y(:).^2); 
    end
end

if nargout > 7
    ssqx = -diff(ssqx./ssqx(1,1));
    ssqy = -diff(ssqy./ssqy(1,1));
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
