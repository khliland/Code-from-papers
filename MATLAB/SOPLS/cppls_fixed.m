function [W, P, T, pow, cc, beta, Q, mX, Maks, shortX, ssq, ssqy] = cppls_fixed(pc, X, Yprim, Yadd, gamma, wt)
% [W, P, T, pow, cc, beta, mX, Maks, shortX, ssq] = cppls_fixed(pc, X, Yprim,
% Yadd, gamma, wt)
%
% -- INPUTS --
% pc     - number of components
% X      - data matrix
% Yprim  - primary response matrix
% -- Optional inputs --
% Yadd   - additional response matrix (use [] when Yadd is absent)
% gamma  - vector of gamma values for powers
% wt     - prior weighting of observations
%
% -- OUTPUTS --
% W      - W-loadings
% P      - P-loadings
% T      - T-scores
% pow    - gamma values for each component
% cc     - canonical correlations
% beta   - regression coefficients
% mX     - centering matrix
% Maks   - maximal values from standard deviation and correlation block
% shortX - vector of X-variables with short norms (after deflation)
% ssq    - variation in the data for each component
%
% -- EXAMPLES --
% CPPLS with addtional responses:
% [W, P, T, pow, cc, beta, mX, Maks, shortX, ssq] = cppls(10, X, Y,
% U, 0, 1);
% CPLS:
% [W, P, T] = cppls(10, X, Y);

[n,m] = size(X);

if nargin < 4
    Yadd = [];
end
if nargin < 5      % Defaults to CPLS if no powers are supplied
    gamma = 0.5;
end
if nargin < 6
    [X, mX] = Center(X);    % Centering of X
    wt = [];                % No weighting of observations
else
    [X, mX] = Center(X,wt); % Weighted centering of X
end
Y = [Yprim Yadd];

% Declaration of variables
W = zeros(m,pc);         % W-loadings
T = zeros(n,pc);         % T-scores
P = zeros(m,pc);         % P-loadings
Q = zeros(size(Yprim,2),pc);
cc     = zeros(pc,1);    % Squared canonical correlations
pow    = ones(pc,1)*0.5; % Powers used to construct the w-s in R
Maks   = zeros(pc,2);    % Largest scalars from the factors in R
orto   = 0;              % orto = 1 indicates orthogolalization of loading weights
shortX = [];             % Vector indicating X variables with short norms
beta   = zeros(m,size(Yprim,2),pc);
if nargout > 9
    ssq  = zeros(pc,3);    % Remaining variation in the data (for variance summary)
    ssqx = sum(X(:).^2);   % Total variation in the data
    ssqy = zeros(pc,1);
    F    = Yprim;
end

for a=1:pc
    if gamma(a)==0.5
        [W(:,a), cc(a,1)] = Ra(X, Y, Yprim, wt);
    else
        [W(:,a), pow(a,1), Maks(a,:), cc(a,1)] = Rb(X, Y, Yprim, wt, gamma(a));
    end
    
    % Make new vectors orthogonal to old ones
    if orto == 1
        W(:,a) = W(:,a) - W(:,1:(a-1))*(W(:,1:(a-1))'*W(:,a));
    end
    W(abs(W(:,a))<eps,a) = 0;                % Removes insignificant values
    W(:,a) = W(:,a)./norm(W(:,a));           % Normalization
    T(:,a) = X*W(:,a);                       % Score vectors
    P(:,a) = (X'*T(:,a))./(T(:,a)'*T(:,a));  % Loadings
    X = X - T(:,a)*P(:,a)';                  % Deflation
    
    % Check and compensate for small norms
    mm = ones(1,n)*abs(X);   % norm1 of all collumns
    r = find(mm < 10^-12);   % small norms
    shortX = [shortX r(~ismember(r,shortX))]; % add new short norms to list
    X(:,shortX) = 0;         % remove collumns of short norms

    if nargout > 5
        Q(:,a) = (Yprim'*T(:,a))./(T(:,a)'*T(:,a));
        beta(:,:,a) =  W(:,1:a)/(P(:,1:a)'*W(:,1:a))*Q(:,1:a)';
    end
    
    if nargout > 9 % Residual X-variance
        ssq(a,1) = sum(X(:).^2)*100/ssqx; 
        F = F - T(:,a)*Q(:,a)';
        ssqy(a,1) = sqrt(mean(F(:).^2));
    end
end

if nargout > 9
    % Compute variance summary
    ssqdif   = zeros(pc,1);
    ssqdif(1,1) = 100 - ssq(1,1);
    for i = 2:pc
        ssqdif(i,1) = -ssq(i,1) + ssq(i-1,1);
    end
    ssq = [(1:pc)' ssqdif(:,1) cumsum(ssqdif(:,1))];
end


%% Ra function (for CPLS excluding powers)
function [w,cc] = Ra(X, Y, Yprim, wt)
W = X'*Y;
[r,A] = ccXY(X*W, Yprim, wt); % Computaion of canonical correlations between
                              % XW and Y with the rows weighted according to wt.
w     = W*A(:,1);             % The optimal loading weight vector
cc    = r(1)^2;               % squared canonical correlation


%% Rb function (for CPPLS including powers)
function [w, pow, maks, cc] = Rb(X, Y, Yprim, wt, gamma)

[C, Sx] = CorrXY(X,Y,wt);    % Matrix of corr(Xj,Yg) and vector of std(Xj)
sng = sign(C);               % Signs of C {-1,0,1}
C = abs(C);                  % Correlation without signs
mS = max(Sx); Sx = Sx./mS;   % Divide by largest value
mC = max(max(C)); C = C./mC; %  -------- || --------
maks = [mS, mC];

% Computation of the best vector of loadings
[w, pow, cc] = lw_bestpar(X, Sx, C, sng, Yprim, wt, gamma);


%% lw_bestpar function
function [w, pow, cc] = lw_bestpar(X, Sx, C, sng, Yprim, wt, gamma)
pow  = gamma;
wt   = sqrt(wt); % Prepare weights for cca


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computation of final w-vectors based on optimization %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if pow==0     % 1 - Variable selection from standard deviation
    Sx(Sx<max(Sx))=0;
    w = Sx;
elseif pow==1 % 3 - Variable selection from correlation
    C(C<max(max(C)))=0;
    w = sum(C,2);
    %     w = sum(sng.*C,2);
else                % 2 - Standard deviation and correlation with powers
    p  = pow;          % Power from optimization
    Sx = Sx.^((1-p)/p);
    W0 = rscale(sng.*(C.^(p/(1-p))),Sx);
    Z = X*W0;                 % Transform X using W
    [~,A] = ccXY(Z,Yprim,wt); % Compoutes canonical correlations between collumns XW and Y with rows weighted according to wt.
    w = W0*A(:,1);            % Optimal loadings
end

cc = f(gamma, X, Sx, C, sng, Yprim, wt);

%%%%%%%%%%%%%%%%%%%%%%%%%
% Optimization function %
%%%%%%%%%%%%%%%%%%%%%%%%%
function c = f(p, X, Sx, C, sng, Yprim, wt)
if p == 0     % 1 - Variable selection from standard deviation
    Sx(Sx<max(Sx))=0;
    W0 = Sx;
elseif p == 1 % 3 - Variable selection from correlation
    C(C<max(max(C)))=0;
    W0 = sum(C,2);
    % W0 = sum(sng.*C,2);
else          % 2 - Standard deviation and correlation with powers
    Sx = Sx.^((1-p)/p);
    W0 = rscale(sng.*(C.^(p/(1-p))),Sx);
end

Z = X*W0;  % Project X onto W0
[r] = ccXY(Z,Yprim,wt); % Computes cannonical correlation between Z and Yprim with rows weighted by wt
c = -r(1)^2;


%% CorrXY function
function [ccxy sdX] = CorrXY(X,Y,wt)
% Computing of correlations between the columns of X and Y
% and standard deviations of X

n = size(X,1);
X = Center(X,wt);
Y = Center(Y,wt);
if ~isempty(wt)
    X = rscale(X,wt);
    Y = rscale(Y,wt);
end
sdX = std(X,1)';
inds = find(sdX==0);
sdX(inds)=1;             % Remove insignificant std.

ccxy = X'*Y./(n*sdX*std(Y,1,1));
sdX(inds)=0;
ccxy(inds,:) = 0;


%% rscale function
function X = rscale(X,d)
% Scaling the rows of the matrix X by the values of the vector d
X = repmat(d,1,size(X,2)).*X;


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


%% Canonical correlations
function [r,A] = ccXY(X,Y,wt)
% Computes the coefficients in canonical variates between collumns of X and Y
[n,p1] = size(X); p2 = size(Y,2);

% Weighting of observations with regards to wt (asumes weighted centering already performed)
if ~isempty(wt)
    X = rscale(X,wt);
    Y = rscale(Y,wt);
end

% Factoring of data by QR decomposition and ellimination of internal linear
% dependencies in X and Y
[Q1,T11,perm1] = qr(X,0);       [Q2,T22,~] = qr(Y,0);
rankX          = sum(abs(diag(T11)) > eps(abs(T11(1)))*max(n,p1));
rankY          = sum(abs(diag(T22)) > eps(abs(T22(1)))*max(n,p2));
if rankX < p1
    Q1 = Q1(:,1:rankX); T11 = T11(1:rankX,1:rankX);
end
if rankY < p2
    Q2 = Q2(:,1:rankY);
end

% Economical computation of canonical coefficients and canonical correlations
d = min(rankX,rankY);
if nargout == 1
    D    = svd(Q1' * Q2,0);
    r    = min(max(D(1:d), 0), 1); % Canonical correlations
else
    [L,D]    = svd(Q1' * Q2,0);
    A        = T11 \ L(:,1:d) * sqrt(n-1);
    % Transform back coefficients to full size and correct order
    A(perm1,:) = [A; zeros(p1-rankX,d)];
    r = min(max(diag(D(1:d)), 0), 1); % Canonical correlations
end