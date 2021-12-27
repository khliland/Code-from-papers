function [W, P, T, pot, ssq, Maks] = pplsda(pc, X, Y, lower, upper, dis, Pi)
% [W, pot, P, T, ssq, Maks] = pplsda(pc, X, Y, lower, upper, dis, Pi)
% 
% -- Inputs --
% X     - the data matrix
% Y     - the respons matrix
% pc    - number of components
% lower - lower bounds for the powers
% upper - upper bounds for the powers
% dis   - discard t/w vectors shorter than dis (optional, default = 0)
% Pi    - prior weighting of groups            (optional)
% 
% -- Outputs --
% W     - W-loadings
% P     - P-loadings
% T     - T-scores
% pot   - gamma values for each component
% ssq   - variation in the data for each component
% Maks  - maximal values from standard deviation and correlation block

[n,m] = size(X);
Yd = dummy(Y);

if nargin < 5
    lower = 0;
    upper = 1;
end
if nargin < 6
    dis = 0;  % Typically 10^-10 if used
end
if nargin < 7
    Pi = sum(Yd)./n;
    ce = [];
else
    ce = Yd*(Pi./(sum(Yd)./n))';
end

% Centering X
X = Center(X,ce);

% Matrices common to all components
Ro = diag((std(Yd,1).*sqrt(Pi)./sum(Yd))); % P in article

% Declaration of variables
W = zeros(m,pc);           % W-loadings
T = zeros(n,pc);           % T-scores
P = zeros(m,pc);           % P-loadings
pot = zeros(pc,1);         % Powers used to construct the w-s in R
ssq = zeros(pc,3);         % Remaining variation in the data
ssqx = sum(sum(X.^2),2)';  % Total variation in the data
Maks = zeros(pc,2);        % Largest scalars from the factors in R
orto = 0;                  % orto = 1 indicates orthogolalization of loading weights

a = 1;
while a <= pc
    [W(:,a), pot(a,1), Maks(a,:)] = R(X, Yd, lower, upper, Ro);%, V, YD);
    k=0;
    if norm(W(:,a)) < dis % Cheks if w is very short
        k=1;
        disp('Short w')
    end
    
    W(:,a) = W(:,a)./max(abs(W(:,a)));       % Stabilization
    % Make new vectors orthogonal to old ones
    if orto == 1
        W(:,a) = W(:,a) - W(:,1:(a-1))*(W(:,1:(a-1))'*W(:,a));
    end
    W(abs(W(:,a))<eps,a) = 0;                % Removes insignificant values
    W(:,a) = W(:,a)./norm(W(:,a));           % Normalization
    T(:,a) = X*W(:,a);                       % Score vectors
    P(:,a) = (X'*T(:,a))/(T(:,a)'*T(:,a));   % Loadings
    X = X - T(:,a)*P(:,a)';                  % Deflation
    ssq(a,1) = (sum(sum(X.^2),2)')*100/ssqx; % Residual X-variance
    
    if norm(T(:,a)) < dis % Cheks if t is very short
        k = 1;
        disp('Short t')
    end
    if k == 0
        a = a+1;	% If neither w nor t is very short, go to next component
    end
end

% Sum of squares for components
ssqdif   = zeros(pc,2);
ssqdif(1,1) = 100 - ssq(1,1);
for i = 2:pc
    ssqdif(i,1) = -ssq(i,1) + ssq(i-1,1);
end
ssq = [(1:pc)' ssqdif(:,1) cumsum(ssqdif(:,1))];


%% R function
function [w, pot, maks] = R(X, Yd, lower, upper, Ro)%, V, YD)

[C, Sx] = CorrXY(X,Yd);       % Matrix of corr(Xj,Yg) and vector of std(Xj)
sng = sign(C);               % Signs of C {-1,0,1}
C = abs(C);                  % Correlation without signs
mS = max(Sx); Sx = Sx./mS;   % Divide by largest value
mC = max(max(C)); C = C./mC; %  -------- || --------
maks = [mS, mC];

% Computation of the best vector of loadings
[w, pot] = lw_bestpar(X, Sx, C, sng, lower, upper, Ro, Yd);%, V, YD);


%% lw_bestpar function
function [w, pot] = lw_bestpar(X, Sx, C, sng, lower, upper, Ro, Yd)%, V, YD)
nOpt = length(lower);   % Number of optimization intervals
pot = zeros(3*nOpt,1);  % Gamma initialization
c = zeros(3*nOpt,1);    % Eigen value initialization

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Use end points and interval (lower,upper) %
% to get eigenvalues to compare             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:nOpt
    c(1+(i-1)*3,1)   = f(lower(i), X, Sx, C, sng, Ro, Yd);%, V, YD);
    pot(1+(i-1)*3,1) = lower(i);
    if(lower(i) ~= upper(i))
        [pot(2+(i-1)*3,1),c(2+(i-1)*3,1)] = fminbnd(@(p) f(p, X, Sx, C, sng, Ro, Yd), lower(i), upper(i));
    end
    c(3+(i-1)*3,1)   = f(upper(i), X, Sx, C, sng, Ro, Yd);%, V, YD);
    pot(3+(i-1)*3,1) = upper(i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computation of final w-vectors based on optimization %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[cm,cmin] = max(-c); % Determine which is more succesful

if pot(cmin)==0      % Variable selection from standard deviation
        Sx(Sx<max(Sx))=0;
        w = Sx;
elseif pot(cmin)==1  % Variable selection from correlation
        C(C<max(max(C)))=0;
        w = sum(C,2);
else                 % Standard deviation and correlation with powers
        p = pot(cmin,1);   % Power from optimization
        Sx = Sx.^((1-p)/p);
        W0 = rscale(sng.*(C.^(p/(1-p))),Sx)*Ro;
        Z = X*W0;          % Transform X into W
        
        % Compute eigenvalues and vectors of T^-1*B
        [A] = ccXY(Z,Yd); % Compoutes canonical correlations between collumns XW and Y with rows weighted according to wt.
        w = W0*A(:,1);          % Optimal loadings
end
pot = pot(cmin,1);


%%%%%%%%%%%%%%%%%%%%%%%%%
% Optimization function %
%%%%%%%%%%%%%%%%%%%%%%%%%
function c = f(p, X, Sx, C, sng, Ro, Yd)%V, YD)
if p == 0     % Variable selection from standard deviation
        Sx(Sx<max(Sx))=0;
        W0 = Sx;
elseif p == 1 % Variable selection from correlation
        C(C<max(max(C)))=0;
        W0 = sum(C,2);
else          % Standard deviation and correlation with powers
        Sx = Sx.^((1-p)/p);
        W0 = rscale(sng.*(C.^(p/(1-p))),Sx)*Ro;
end

Z = X*W0;  % Transform X into W0
[A,B,r] = ccXY(Z,Yd); % Computes cannonical correaltion between Z and Y
c = -r(1)^2;


%% CorrXY function
function [ccxy sdX] = CorrXY(X,Y)
% Computing of correlations between the columns of X and Y
% and standard deviations of X

n = size(X,1);
Y = (Y - ones(n,1)*mean(Y));
X = (X - ones(n,1)*mean(X));
sdX = std(X,1)';
inds = find(sdX==0);
sdX(inds)=1;      % Avoid zero division

ccxy = X'*Y./(n*sdX*std(Y,1,1));
sdX(inds)=0;      % Remove insignificant std. dev.
ccxy(inds,:) = 0; % Remove insignificant corr.


%% dummy function
function dum = dummy(X)
% This function mostly does the same as dummyvar(X)

n = size(X,1);
p = max(X);
dum = zeros(n,p);
for i=1:n
    dum(i,X(i))=1;
end


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

% Centering of X, similar to: X = X-ones(n,1)*mX;
X = X-repmat(mX,n,1);


%% Canonical correlations
function [A,B,r] = ccXY(X,Y)
% Computes the coefficients in canonical variates between collumns of X and Y
[n,p1] = size(X); p2 = size(Y,2);

% Factoring of data by QR decomposition and ellimination of internal linear
% dependencies in X and Y
[Q1,T11,perm1] = qr(X,0);       [Q2,T22,perm2] = qr(Y,0);
rankX          = sum(abs(diag(T11)) > eps(abs(T11(1)))*max(n,p1));
rankY          = sum(abs(diag(T22)) > eps(abs(T22(1)))*max(n,p2));
if rankX < p1
    Q1 = Q1(:,1:rankX); T11 = T11(1:rankX,1:rankX);
end
if rankY < p2
    Q2 = Q2(:,1:rankY); T22 = T22(1:rankY,1:rankY);
end

% Economical computation of canonical coefficients and canonical correlations
d = min(rankX,rankY);
[L,D,M]    = svd(Q1' * Q2,0);
A          = T11 \ L(:,1:d) * sqrt(n-1);
B          = T22 \ M(:,1:d) * sqrt(n-1);
r          = min(max(diag(D(:,1:d)), 0), 1); % Canonical correlations
% Transform back coefficients to full size and correct order
A(perm1,:) = [A; zeros(p1-rankX,d)];
B(perm2,:) = [B; zeros(p2-rankY,d)];

