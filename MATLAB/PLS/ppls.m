function [W, P, T, pot, beta, mX] = ppls(pc, X, Y, lower, upper)
% X - Xdata, Y - Yvector, pc - the number of components to be extracted
% lower - use 0 or larger, upper - use 1 or smaller (0 and 1) defaults.
% Algorithm returns regression coeffs, W, P and T according to
% notational convetions, mean of the X-vatiables, mean of the Y-vector.

[n,m] = size(X);
% Centering of X & Y:
mX = mean(X);
X = X-ones(n,1)*mX;
mY = mean(Y);
Y = Y-mY;
if nargin == 3
    lower = 0;
    upper = 1;
end

% Declaration of variables
W = zeros(m,pc);          % Loading weights
T = zeros(n,pc);          % T-scores
P = zeros(m,pc);          % P-loadings
Q = zeros(1,pc);          % Q-loadings (trivial here)
beta = zeros(m,pc);       % Regression coefficients for models of 1 to pc components
pot = zeros(pc,1);        % For powers from calculations of w-s in R
orto = 0;                 % <> 0 indicates orthogolalization of loading weights

for a = 1:pc              % Notation corresponding to frame 3.4 of Martens & Næs
    [w, pot(a)] = R(X, Y, W(:,1:a), lower, upper, orto);
    % R returns loadingweights (w) and the best corresponding parameter (pot(a)) according to the chosen strategy
%     W(:,a) = W(:,a)./max(abs(W(:,a)));       % Stabilization
    W(:,a) = w/norm(w);
%     ind1 = find(abs(W(:,a)) >= eps);
%     ind2 = find(abs(W(:,a)) <  eps);
%     W(ind2,a)=0;
%     t = X(:,ind1)*W(ind1,a);
    t = X*W(:,a);
    nt = t/(t'*t);
    T(:,a) = t;
    P(:,a) = X'*nt;
    Q(a) = Y'*nt;
    X = X-T(:,a)*P(:,a)';
    Y = Y-T(:,a)*Q(a);
    if nargout > 4
        beta(:,a) = W(: ,1:a) * pinv(P(:,1:a)'*W(: ,1:a)) * Q(1:a)';
    end
end

%%
function [w, pot] = R(X, Y, W, lower, upper, orto)
% [w, pot] = R(X, Y, W, orto, lower, upper) - The function preprocessing based on
% the X and Y data. The a function (lw_bestpar) for caluculation of the optimal loading
% weight based on the requested parameter interval [lower, upper] is called.
% The solution [w, pot] consisting of a loading weight (w) and the corresponding
% parameter value (pot) is returned. Optionally orthogonalization of w with
% respect to earlier loadnig weights in W executed.

[u,v] = CorrXY(X, Y);   % Correlations between the X-variables and Y and standard deviations
sng = sign(u);          % Sign of the correlations

u = abs(u);             % Absolute values of the correlations

[mu iu] = max(u); [mv iv] = max(v); % 'iu' is the index of the X-variable most correlated to Y
% 'iv' is the index of the X-variable with the largest standard deviation

u = u/mu; v = v/mv;     % Scaled absoulte values of the correlations and standard deviations
% (to assure max(u) = max(v) = 1). This is tric to
% stabelize computations inside 'lw_bestpar'.

[w, pot] = lw_bestpar(X, Y, u, v, sng, iu, lower, upper); % Computation of the best loading
% weight and corresponding parameter value

if orto ~= 0            % Optional: To remove the W-directions accounted for by earlier
    w = w-W*W'*w;       % loading weights if indicatied by "orto"
end;

%%
function [w, pot] = lw_bestpar(X, Y, u, v, sng, iu, lower, upper)
% This function is called by the function 'R' and it finds the parameter pot
% in the interval [lower, upper] and the corresponding loading weight w that
% (in absolute value) maximizes the correlation between Y and Xw. The
% optimization is handeled by calling the MATLAB system-function 'fminbnd' to
% minimize the function 'f(p, X, Y, sng, u, v)'.
% 'fminbnd' implements a one-variable optimization by the golden section search
% and parabolic interpolation method (see Brent, Richard. P., Algorithms for
% Minimization without Derivatives, Prentice-Hall, Englewood Cliffs, New
% Jersey, 1973).

nOpt = length(lower);
pot = zeros(3*nOpt,1);
c = zeros(3*nOpt,1);

for i=1:nOpt
    c(1+(i-1)*3,1)   = f(lower(i), X, Y, sng, u, v);
    pot(1+(i-1)*3,1) = lower(i);
    if(lower(i) ~= upper(i))
        [pot(2+(i-1)*3,1),c(2+(i-1)*3,1)] = fminbnd(@(p) f(p, X, Y, sng, u, v), lower(i), upper(i));
    end
    c(3+(i-1)*3,1)   = f(upper(i), X, Y, sng, u, v);
    pot(3+(i-1)*3,1) = upper(i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computation of final w-vectors based on optimization %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[cm,cmin] = max(-c); % Determine which is more succesful
% disp(cm);

if pot(cmin)==0     % 1 - Variable selection from standard deviation
    [mv iv] = max(v);
    w = zeros(length(v),1);
    w(iv) = 1;
elseif pot(cmin)==1 % 3 - Variable selection from correlation
    w = zeros(length(u),1);
    w(iu) = 1;
else                % 2 - Standard deviation and correlation with powers
    p = pot(cmin,1); % Power from optimization
    w = sng.*(u.^(p/(1-p)).*v.^((1-p)/p)); % The correspondnig loading weights are calculated.
end

pot = pot(cmin,1);


%% Maximization function
function c = f(p, X, Y, sng, u, v)
if p == 1
    [ma, maxid] = max(u);
    w = zeros(length(u),1);
    w(maxid) = 1;
elseif p == 0
    [ma, maxid] = max(v);
    w = zeros(length(v),1);
    w(maxid) = 1;
else
    w = sng.*(u.^(p/(1-p)).*v.^((1-p)/p)); % The expression for a valid loading weight
end

% Maximization of the correlation between (Xw and Y)is equivalent to minimization of:
c = -CorrXY(X*w,Y)^2;
% c = -sum((X*w-Y).^2);


%% CorrXY function
function [ccxy sdX] = CorrXY(X,Y)
%  Computing of correlations between the columns of X and Y

n = size(X,1);
Y = (Y - ones(n,1)*mean(Y));
X = (X - ones(n,1)*mean(X));
sdX = std(X,1)';
inds = find(sdX==0);
sdX(inds)=1;             % Remove insignificant std.

ccxy = X'*Y./(n*sdX*std(Y,1,1));
sdX(inds)=0;
ccxy(inds,:) = 0;

