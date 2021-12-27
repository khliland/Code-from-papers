function [W, P, T, pow, cc, beta, Q, mX, Maks, shortX, ssqx, ssqy] = cppls(pc, X, Yprim, Yadd, lower, upper, wt, conf)
% [W, P, T, pow, cc, beta, mX, Maks, shortX, ssqx, ssqy] = cppls(pc, X, Yprim,
% Yadd, lower, upper, wt)
%
% -- INPUTS --
% pc     - number of components
% X      - data matrix
% Yprim  - primary response matrix
% -- Optional inputs --
% Yadd   - additional response matrix (use [] when Yadd is absent)
% lower  - lower bounds for the powers
% upper  - upper bounds for the powers
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
% ssqx   - variation in the X data for each component
% ssqy   - variation in the y data for each component
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
    lower = 0.5;
    upper = 0.5;
end
if nargin < 7
    [X, mX] = Center(X);    % Centering of X
    wt = [];                % No weighting of observations
else
    [X, mX] = Center(X,wt); % Weighted centering of X
end
if nargin < 8
    conf = 0;
end
Y  = [Yprim Yadd];
Yc = Center(Yprim);

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
    ssqx = [sum(X(:).^2);zeros(pc,1)];   % Total variation in the data
    ssqy = [sum(Yc(:).^2);zeros(pc,1)];  % Total variation in the data
end

for a=1:pc
    if length(lower)==1 && lower == 0.5 && length(upper)==1 && upper == 0.5
        [W(:,a), cc(a,1)] = Ra(X, Y, Yprim, wt);
    else
        [W(:,a), pow(a,1), Maks(a,:), cc(a,1)] = Rb(X, Y, Yprim, wt, lower, upper);
    end
    
    if conf ~= 0
        tdf = (X*W(:,a)).^2;
        df = ceil(sum(tdf)./max(tdf));
        doPlot = 0; distribution = 1; quant = 0; weightW = 0; sym = 0;
        [~,wWeights] = normtrunc(W(:,a),doPlot,a,df,distribution,quant,weightW,conf,sym);
        wWeights(wWeights<eps) = 0;
        W(:,a)=W(:,a).*wWeights;
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
    Q(:,a) = (Yprim'*T(:,a))./(T(:,a)'*T(:,a));
%     Yc = Yc - T(:,a)*Q(:,a)';                % Deflation
%     Yprim = Yc;
    
    % Check and compensate for small norms
    mm = ones(1,n)*abs(X);   % norm1 of all collumns
    r = find(mm < 10^-12);   % small norms
    shortX = [shortX r(~ismember(r,shortX))]; % add new short norms to list
    X(:,shortX) = 0;         % remove collumns of short norms

    if nargout > 5
        beta(:,:,a) =  W(:,1:a)/(P(:,1:a)'*W(:,1:a))*Q(:,1:a)';
    end
    
    if nargout > 9 % Residual X-variance
        ssqx(a+1,1) = sum(X(:).^2); 
        Yc = Yc - T(:,a)*Q(:,a)';                % Deflation
        ssqy(a+1,1) = sum(Yc(:).^2); 
    end
end

if nargout > 9
    ssqx = -diff(ssqx./ssqx(1,1));
    ssqy = -diff(ssqy./ssqy(1,1));
end


%% Ra function (for CPLS excluding powers)
function [w,cc] = Ra(X, Y, Yprim, wt)
W = X'*Y;
[r,A] = ccXY(X*W, Yprim, wt); % Computaion of canonical correlations between
                              % XW and Y with the rows weighted according to wt.
w     = W*A(:,1);             % The optimal loading weight vector
cc    = r(1)^2;               % squared canonical correlation


%% Rb function (for CPPLS including powers)
function [w, pow, maks, cc] = Rb(X, Y, Yprim, wt, lower, upper)

[C, Sx] = CorrXY(X,Y,wt);    % Matrix of corr(Xj,Yg) and vector of std(Xj)
sng = sign(C);               % Signs of C {-1,0,1}
C = abs(C);                  % Correlation without signs
mS = max(Sx); Sx = Sx./mS;   % Divide by largest value
mC = max(max(C)); C = C./mC; %  -------- || --------
maks = [mS, mC];

% Computation of the best vector of loadings
[w, pow, cc] = lw_bestpar(X, Sx, C, sng, Yprim, wt, lower, upper);


%% lw_bestpar function
function [w, pow, cc] = lw_bestpar(X, Sx, C, sng, Yprim, wt, lower, upper)
nOpt = length(lower);
pow  = zeros(3*nOpt,1);
c    = zeros(3*nOpt,1);
wt   = sqrt(wt); % Prepare weights for cca

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Use end points and interval (lower,upper) %
% to get eigenvalues to compare             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:nOpt
    c(1+(i-1)*3,1)   = f(lower(i), X, Sx, C, sng, Yprim, wt);
    pow(1+(i-1)*3,1) = lower(i);
    if(lower(i) ~= upper(i))
        [pow(2+(i-1)*3,1),c(2+(i-1)*3,1)] = fminbnd(@(p) f(p, X, Sx, C, sng, Yprim, wt), lower(i), upper(i));
    end
    c(3+(i-1)*3,1)   = f(upper(i), X, Sx, C, sng, Yprim, wt);
    pow(3+(i-1)*3,1) = upper(i);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computation of final w-vectors based on optimization %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[cm,cmin] = max(-c); % Determine which is more succesful

if pow(cmin)==0     % 1 - Variable selection from standard deviation
    Sx(Sx<max(Sx))=0;
    w = Sx;
elseif pow(cmin)==1 % 3 - Variable selection from correlation
    C(C<max(max(C)))=0;
    w = sum(C,2);
    %     w = sum(sng.*C,2);
else                % 2 - Standard deviation and correlation with powers
    p  = pow(cmin,1);          % Power from optimization
    Sx = Sx.^((1-p)/p);
    W0 = rscale(sng.*(C.^(p/(1-p))),Sx);
    Z = X*W0;                 % Transform X using W
    [~,A] = ccXY(Z,Yprim,wt); % Compoutes canonical correlations between collumns XW and Y with rows weighted according to wt.
    w = W0*A(:,1);            % Optimal loadings
end

pow = pow(cmin,1);
cc = cm;

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
% X = repmat(d,1,size(X,2)).*X;
X = bsxfun(@times,X,d);


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