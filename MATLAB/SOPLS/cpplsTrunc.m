function [W, P, T, pow, cc, beta, mX, Maks, shortX, ssq, ssqy] = cpplsTrunc(pc, X, Yprim, Yadd, lower, upper, wt)
% [W, P, T, pow, cc, beta, mX, Maks, shortX, ssq] = cppls(pc, X, Yprim,
% Yadd, lower, upper, powerType, wt)
%
% -- INPUTS --
% pc     - number of components
% X      - data matrix
% Yprim  - primary response matrix
% -- Optional inputs --
% Yadd   - additional response matrix (use [] when Yadd is absent)
% lower  - lower bounds for the powers
% upper  - upper bounds for the powers
% powerType - ordinary or truncated powers
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

powerType = 0;
[n,m] = size(X);

if nargin < 4
    Yadd = [];
end
if nargin < 6      % Defaults to CPLS if no powers are supplied
    lower = 0.5;
    upper = 0.5;
end

if nargin < 8
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
    ssqyT = sum(F(:).^2);
end

for a=1:pc
    if length(lower)==1 && lower == 0.5 && length(upper)==1 && upper == 0.5
        [W(:,a), cc(a,1)] = Ra(X, Y, Yprim, wt);
    else
        [W(:,a), pow(a,1), Maks(a,:), cc(a,1)] = Rb(X, Y, Yprim, wt, lower, upper, powerType);
    end
    
    distribution = 0;
    conf = 0.95;
    sym  = 0;
    weightW = 0;
    quant   = 0;
    doPlot  = 0;
    if distribution == 0 || distribution == 3
        tdf = (X*W(:,a)).^2;
        df = ceil(sum(tdf)./max(tdf));
    else
        df = 0;
    end
    [~,wWeights] = normtrunc(W(:,a),doPlot,a,df,distribution,quant,weightW,conf,sym);
    wWeights(wWeights<eps) = 0;
    W(:,a)=W(:,a).*wWeights;
    
    % Make new vectors orthogonal to old ones
    if orto == 1
        W(:,a) = W(:,a) - W(:,1:(a-1))*(W(:,1:(a-1))'*W(:,a));
    end
    W(abs(W(:,a))<eps,a) = 0;                % Removes insignificant values
    W(:,a) = W(:,a)./norm(W(:,a));           % Normalization
    T(:,a) = X*W(:,a);                       % Score vectors
    P(:,a) = (X'*T(:,a))./(T(:,a)'*T(:,a));  % Loadings
    X = X - T(:,a)*P(:,a)';                  % Deflation
%     Q(:,a) = (Yprim'*T(:,a))./(T(:,a)'*T(:,a));
%     Yprim = Yprim - T(:,a)*Q(:,a)';                  % Deflation
%     Y = Y - T(:,a)*Q(:,a)';                  % Deflation
    
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
    
%     ssqy = [(1:pc)' ssqdif(:,1) cumsum(ssqdif(:,1))];
end


%% Ra function (for CPLS excluding powers)
function [w,cc] = Ra(X, Y, Yprim, wt)
W = X'*Y;
[r,A] = ccXY(X*W, Yprim, wt); % Computaion of canonical correlations between
                              % XW and Y with the rows weighted according to wt.
w     = W*A(:,1);             % The optimal loading weight vector
cc    = r(1)^2;               % squared canonical correlation


%% Rb function (for CPPLS including powers)
function [w, pow, maks, cc] = Rb(X, Y, Yprim, wt, lower, upper, powerType)

[C, Sx] = CorrXY(X,Y,wt);    % Matrix of corr(Xj,Yg) and vector of std(Xj)
sng = sign(C);               % Signs of C {-1,0,1}
C = abs(C);                  % Correlation without signs
mS = max(Sx); Sx = Sx./mS;   % Divide by largest value
mC = max(max(C)); C = C./mC; %  -------- || --------
maks = [mS, mC];

% Computation of the best vector of loadings
[w, pow, cc] = lw_bestpar(X, Sx, C, sng, Yprim, wt, lower, upper, powerType);


%% lw_bestpar function
function [w, pow, cc] = lw_bestpar(X, Sx, C, sng, Yprim, wt, lower, upper, powerType)
nOpt = length(lower);
pow  = zeros(3*nOpt,1);
c    = zeros(3*nOpt,1);
wt   = sqrt(wt); % Prepare weights for cca
medC = abs(sng.*C-ones(size(Sx,1),1)*median(sng.*C));
medC = medC./(ones(size(Sx,1),1)*max(medC));
medS = abs(Sx-median(Sx));
medS = medS./max(medS);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Use end points and interval (lower,upper) %
% to get eigenvalues to compare             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:nOpt
    c(1+(i-1)*3,1)   = f(lower(i), X, Sx, C, sng, Yprim, wt, medS, medC, powerType);
    pow(1+(i-1)*3,1) = lower(i);
    if(lower(i) ~= upper(i))
        [pow(2+(i-1)*3,1),c(2+(i-1)*3,1)] = fminbnd(@(p) f(p, X, Sx, C, sng, Yprim, wt, medS, medC, powerType), lower(i), upper(i));
    end
    c(3+(i-1)*3,1)   = f(upper(i), X, Sx, C, sng, Yprim, wt, medS, medC, powerType);
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
    if powerType == 0 % Original power handling
        Sx = Sx.^((1-p)/p);
        W0 = rscale(sng.*(C.^(p/(1-p))),Sx);
    else % Truncation type power handling
        ps = (1-p)/p;
        if(ps<1)
            Sx = Sx.^ps;
        else
            Sx(medS<(1-2*p)) = 0;
        end
        pc = p/(1-p);
        if(pc<1)
            W0 = rscale(sng.*(C.^pc),Sx);
        else
            C(medC<(2*p-1)) = 0;
            W0 = rscale(sng.*C,Sx);
        end
    end
    Z = X*W0;                 % Transform X using W
    [~,A] = ccXY(Z,Yprim,wt); % Compoutes canonical correlations between collumns XW and Y with rows weighted according to wt.
    w = W0*A(:,1);            % Optimal loadings
end

pow = pow(cmin,1);
cc = cm;

%%%%%%%%%%%%%%%%%%%%%%%%%
% Optimization function %
%%%%%%%%%%%%%%%%%%%%%%%%%
function c = f(p, X, Sx, C, sng, Yprim, wt, medS, medC, powerType)
if p == 0     % 1 - Variable selection from standard deviation
    Sx(Sx<max(Sx))=0;
    W0 = Sx;
elseif p == 1 % 3 - Variable selection from correlation
    C(C<max(max(C)))=0;
    W0 = sum(C,2);
    % W0 = sum(sng.*C,2);
else          % 2 - Standard deviation and correlation with powers
    if powerType == 0 % Original power handling
        Sx = Sx.^((1-p)/p);
        W0 = rscale(sng.*(C.^(p/(1-p))),Sx);
    else % Truncation type power handling
        ps = (1-p)/p;
        if(ps<1)
            Sx = Sx.^ps;
        else
            Sx(medS<(1-2*p)) = 0;
        end
        pc = p/(1-p);
        if(pc<1)
            W0 = rscale(sng.*(C.^pc),Sx);
        else
            C(medC<(2*p-1)) = 0;
            W0 = rscale(sng.*C,Sx);
        end
    end
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

%% Truncation part
function [InLiers, wWeights] = normtrunc(x,PlotIt,a,df,distribution,quant,weight,conf,sym)
% weight: Weight by inverse cummulative norm/t = 1, Sharp cut-off = 0
% distribution: Empirical = 3, normal approximation = 2, student-t = 1

[med,sx,i,n] = Median(x);
if(abs(sx(1))>abs(sx(end)))
    max_out = i(1);
else
    max_out = i(end);
end
wWeights = ones(n,1);
dev    = max(sx(end)-med, med-sx(1)); % Maximum deviation from the median
% Minimum deviation at least 6th closest point around median or 1% of the points
minDev = max(abs(med - sx(ceil(n/2) + (-max(ceil(n/200),3):max(ceil(n/200),3) ))));
tolX    = max(dev*10^-4,eps);
rmseMax = max(dev*10^-2,eps);

if conf > 0  % Contrast without replicate (RV Lenth 1989)
    if conf == 0
        alpha = 0.975;
    else
        alpha = conf+(1-conf)/2;
    end
    if sym == 1
        c   = abs(x); s0  = 1.5*median(c);
        PSE = 1.5*median(c(c<2.5*s0));
        offset = PSE;
        if df == 0
            offsets = [1,1]*tinv(alpha,1000)*PSE;
        else
            offsets = [1,1]*tinv(alpha,ceil(df/3))*PSE;
        end
    else
        offsets = [1,1];
        for j = 1:2
            if j==1
                c   = abs(x(x>=0));
            else
                c   = abs(x(x<0));
            end
            s0  = 1.5*median(c);
            PSE = 1.5*median(c(c<2.5*s0));
            offset = PSE;
            if df == 0
                offsets(j) = tinv(alpha,1000)*PSE;
            else
                offsets(j) = tinv(alpha,ceil(df/3))*PSE;
            end
        end
        offset = min(offsets);
    end
elseif quant > 0 % Adapt straight line in quantile plot
    eprob = ((1:n)' - 0.5)./n;
    if distribution == 0
        y  = tinv(eprob,df);
    elseif distribution == 1
        y  = norminv(eprob,0,1);
    elseif distribution == 2
        y = linspace(0,1,n);
    end
    q1x = prctile(x,50-100*quant);       q3x = prctile(x,50+100*quant);
    q1y = prctile(y,50-100*quant);       q3y = prctile(y,50+100*quant);
    dx = q3x - q1x;            dy = q3y - q1y;
    slope = dy./dx;
    centerx = (q1x + q3x)/2;   centery = (q1y + q3y)/2;
    offset = fminbnd(@(tr) minQQ(tr,0,0,sx,y,[centerx,centery],slope,n),minDev,dev, optimset('TolX',tolX)); % Symmetrically
    if sym ~= 1
        offsetL = fminbnd(@(tr) minQQ(tr,offset,-1,sx,y,[centerx,centery],slope,n),offset-eps,dev, optimset('TolX',tolX)); % Symmetrically
        offsetR = fminbnd(@(tr) minQQ(tr,offsetL,1,sx,y,[centerx,centery],slope,n),offset-eps,dev, optimset('TolX',tolX)); % Symmetrically
        offsets = fminsearch(@(tr) minQQ(tr,0,0,sx,y,[centerx,centery],slope,n),[offsetL,offsetR], optimset('TolX',tolX)); % Asymmetrically
    else
        offsets = [1,1]*offset;
    end
else % Direct fit of distribution to data
    if distribution == 1 % Trim for student-t
        offset = fminbnd(@(tr) minT(tr,sx,med,df),minDev,dev, optimset('TolX',tolX)); % Symmetrically
        if sym ~= 1
            offsets = fminsearch(@(tr) minT2(tr,sx,med,df),[offset,offset], optimset('TolX',tolX)); % Asymmetrically
        else
            offsets = [1,1]*offset;
        end
    elseif distribution == 2 % Trim for normality
        offset = fminbnd(@(tr) minNorm(tr,sx,med),minDev,dev, optimset('TolX',tolX)); % Symmetrically
        if sym ~= 1
            offsets = fminsearch(@(tr) minNorm2(tr,sx,med),[offset,offset], optimset('TolX',tolX)); % Asymmetrically
        else
            offsets = [1,1]*offset;
        end
    elseif distribution == 3 % Empirical distribution
        offset = fminbnd(@(tr) minE(tr,sx,med,rmseMax),minDev,dev, optimset('TolX',tolX)); % Symmetrically
        if sym ~= 1
            offsets = fminsearch(@(tr) minE2(tr,sx,med,rmseMax),[offset,offset], optimset('TolX',tolX)); % Asymmetrically
        else
            offsets = [1,1]*offset;
        end     
    end
end

% The variables found to be normally distributed after trimming
if PlotIt
    InLiers = find((sx<(med+max(offset,offsets(2)))) & (sx>(med-max(offset,offsets(1)))));
    InLiers(InLiers==max_out)=[]; % At least one point outside
    plotIt(x,sx,i,InLiers,a,distribution,conf,quant,df)
end
InLiers = (x<(med+max(offset,offsets(2)))) & (x>(med-max(offset,offsets(1))));
InLiers(max_out) = false; % At least one point outside

if conf > 0 && distribution == 2 % Use symmetrical confidence interval based on inliers
    xx = x(InLiers);
    X = norminv((1-conf)/2+[0,conf],mean(xx),std(xx));
    InLiers = (x<X(2)) & (x>X(1));
end

if weight > 0 % Weights from normal distribution
    xTr = x(x<(med+offsets(2)) & x>(med-offsets(1))); % Trim asymmetrically around the median
    sd = std(xTr);
    if distribution == 1
        wWeights = normpdf(x,med,sd);
    else
        wWeights = 1-2*abs(tcdf((x-med)./sd,df)-0.5);
    end
    %         wWeights = tpdf((x-med)./sd,df);
    wWeights = wWeights - min(wWeights);
    wWeights = 1 - wWeights./max(wWeights);
    if weight ~= 1 % Parameterised sharpening of weights towards cut-off
        m1 = max(wWeights(InLiers));
        m2 = min(wWeights(~InLiers));
        weight = 1-weight/2;
        w = weight/(1-weight);
        if m1 > 0
            wWeights(InLiers) = ((wWeights(InLiers)./m1).^w).*m1;
        end
        if m2 < 1
            wWeights(~InLiers) = 1-(((1-wWeights(~InLiers))./(1-m2)).^w).*(1-m2);
        end
    end
else % Sharp cut-off, weights = 0 or 1
    wWeights(InLiers)  = 0;
    wWeights(~InLiers) = 1;
end
InLiers = find(InLiers);

%% Minimise the difference between an inverse normal distribution
% and a trimmed, sorted x (symmetrically)
function msd = minNorm(tr,x,med)
x = x(x<(med+tr) & x>(med-tr)); % Trim symmetrically around the median
n1 = numel(x);
sd = std(x);
n = numel(x);
msd = mean((((-sqrt(2)*sd).*erfcinv(linspace(2/n,2-2/n,n)'))+med-x).^2)*(n1/n);


%% Minimise the difference between an inverse normal distribution
% and a trimmed, sorted x (asymmetrically)
function mse = minNorm2(tr,x,med)
n1 = numel(x);
x = x(x<(med+tr(2)) & x>(med-tr(1))); % Trim asymmetrically around the median
sd = std(x);
n = numel(x);
mse = mean((((-sqrt(2)*sd).*erfcinv(linspace(2/n,2-2/n,n)'))+med-x).^2)*(n1/n);


%% Minimise the difference between an inverse student-t distribution
% and a trimmed, sorted x (symmetrically)
function msd = minT(tr,x,med,df)
n1 = numel(x);
x = x(x<(med+tr) & x>(med-tr)); % Trim symmetrically around the median
sd = std(x);
n = numel(x);
y = tinv(linspace(1/n,1-1/n,n),df)*sd+med;
msd = mean((x-y').^2)*(n1/n);


%% Minimise the difference between an inverse student-t distribution
% and a trimmed, sorted x (asymmetrically)
function mse = minT2(tr,x,med,df)
n1 = numel(x);
x = x(x<(med+tr(2)) & x>(med-tr(1))); % Trim asymmetrically around the median
sd = std(x);
n = numel(x);
y = tinv(linspace(1/n,1-1/n,n),df)*sd+med;
mse = mean((x-y').^2)*(n1/n);


%% Minimise the difference between a straight line
% and the empirical distribution (symmetrically)
function mse = minE(tr,x,med,rmseMax)
x = x(x<(med+tr) & x>(med-tr)); % Trim symmetrically around the median
n = numel(x);
y = linspace(x(1),x(end),n);
mse = (sqrt(mean((x-y').^2))-rmseMax)^2;


%% Minimise the difference between a straight line
% and the empirical distribution (asymmetrically)
function mse = minE2(tr,x,med,rmseMax)
x = x(x<(med+tr(2)) & x>(med-tr(1))); % Trim asymmetrically around the median
n = numel(x);
y = linspace(x(1),x(end),n);
mse = (sqrt(mean((x-y').^2))-rmseMax)^2;


%% Minimise the difference between a quartile line
% and the distribution (asymmetrically)
function mse = minQQ(tr,trF,direct,sx,y,centers,slope,n1)
if direct == 0
    interv = tr.*[-1 1] + centers(1);       % Search both directions
elseif direct == -1
    interv = [tr trF].*[-1 1] + centers(1); % Search to the left
else
    interv = [trF tr].*[-1 1] + centers(1); % Search to the right
end
Y = centers(2) + slope.*(interv-centers(1));
y = y(sx>interv(1) & sx<interv(2));
x = sx(sx>interv(1) & sx<interv(2));
n = length(x);
yi = interp1(interv, Y, x);
mse = mean(abs(yi-y).^2)*n1/n;


%% Median
function [med,sx,i,n] = Median(x)
[sx,i] = sort(x);
n = length(x);
if mod(n,2) == 0
    med = (sx(n/2) + sx(n/2+1)) / 2;
else
    med = sx((n+1)/2);
end


%% Normal qq-plot
function plotIt(x,sx,i,InLiers,a,distribution,conf,quant,df)
n = length(x);

Ind=(1:n)';
sInd=Ind(i);

eprob = ((1:n)' - 0.5)./n;
if distribution == 2 || conf > 0 || quant > 0
    y  = norminv(eprob,0,1);
elseif distribution == 3
    y = linspace(0,1,n);
else
    y  = tinv(eprob,df);
end
q1x = prctile(x,25);       q3x = prctile(x,75);
q1y = prctile(y,25);       q3y = prctile(y,75);
dx = q3x - q1x;            dy = q3y - q1y;
slope = dy./dx;
centerx = (q1x + q3x)/2;   centery = (q1y + q3y)/2;
maxx = max(x);             minx = min(x);
maxy = centery + slope.*(maxx - centerx);
miny = centery - slope.*(centerx - minx);
mx = [minx; maxx];         my = [miny; maxy];

input('Press enter to confirm change', 's')
hold off
plot(sx,y,'.')
axis tight
axis square
hold on
plot(mx,my,'-.r');
plot(x(sInd(InLiers)),y(Ind(InLiers)),'ok')
ylabel('Distribution')
xlabel('Sorted loading weights')
shg

if distribution == 2 || conf > 0 || quant
    title([' Normal distribution plot for component # ',num2str(a)])
elseif distribution == 3
    title([' Empirical distribution plot for component # ',num2str(a)])
elseif distribution == 1
    title([' Student-t (' num2str(df) ' df) distribution plot for component # ',num2str(a)])
else
    title([' Student-t (' num2str(df) ' df) distribution plot for component # ',num2str(a)])
end
% pause(1)
