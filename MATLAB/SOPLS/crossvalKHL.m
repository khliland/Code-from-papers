function [Yhcv, Ycv, gfit, minis, more] = crossvalKHL(method, ncomp, X, Y, varargin)
% # Required input
%   method: 'pls', 'pls2', 'ppls', 'plsda', 'pplsda' or 'cppls'
%   ncomp : number of components
%   X     : data
%   Y     : response
%
% # Optional input
%   cvseg : cross-validation type and number of segments
%           'cvseg', segm, type
%           type: 1 = "leave-one-out", 2 = random
%                 3 = consecutive, 4 = interleaved, 5 = supplied
%   condis: continuous - 'c' or discrete - 'd' responses
%           'condis', {'c','d','c'}
%           - pls, pls2, ppls and cppls defaults to continous
%           - plsda and pplsda defaults to discrete
%   gamma : lower and upper limits of power parameter gamma
%           'gamma', lower, upper
%   Ysec  : secondary response
%           'Ysec', Ysec
%   disc  : discriminant method (lda, qda)
%           'disc', disc
%   trans : transformation of response (cells of exprs., e.g. {'y.^2'})
%           'trans', transformation, retransformation
%   wt    : object weights
%           'wt', wt
%   prior : prior weighting of groups and/or objects [cell(1,#responses)]
%           'prior', prior, obj_group
%           obj_group: 0 = group weights
%                      1 = object weights (single response)
%                      2 = both           ( ---- || ----  )
%   post  : store posterior probabilities generated by LDA/QDA
%   loadW : store vectors of loading weights for each CV segment
%   progress: progress indicator (default = 1, Yes)
%           'progress', progress
%
% # Output
%   Yhcv  : cross-validated responses
%   Ycv   : original responses in same order as Yhcv
%   gfit  : RMSEP for continous responses, proportion of incorrectly
%           classified for discrete responses
%   minis : index and value of minimum gfit and highest gfit not
%           significantly larger than minimum gfit
%   more  : a structure depending on input choices
%           gam   : gamma values from optimizations ordered like Yhcv
%           post  : posterior values from lda/qda ordered like Yhcv
%           loadW : vectors of loading weights for each CV segment
%           order : original order of response
%
% # Example 1
%   [Yhcv, Ycv, gfit, minis] = crossval('cppls', 10, X, Y);
%   - also implies: 'cvseg', 1, size(Y,1), 'gamma', 0.5, 0.5, ...
%                   'Ysec', [], 'disc', 'lda'
%
% # Example 2
%   [Yhcv, Ycv, gfit] = crossval('cppls', 10, X, Y(:,1), 'cvseg', 10, 3,...
%        'gamma', 0, 1, 'Ysec', Y(:,2), 'condis', 'd', 'disc', 'qda');


%% Attributes and preparations
[nx, px] = size(X);
[ny, py] = size(Y);
Yorder   = zeros(ny,1);;


%% Collect optional arguments and set defaults
% CV segments
optcv = find(strcmp('cvseg',varargin)==1);
if isempty(optcv)
    [cv, cvsegm] = cvseg(size(Y,1),size(Y,1),1);
else
    if varargin{optcv+2} == 5
        cv = varargin{optcv+1};
        cvsegm = length(unique(cv));
    else
        [cv, cvsegm] = cvseg(size(Y,1),varargin{optcv+1},varargin{optcv+2});
    end
end

% Continuous or discrete response
optcd = find(strcmp('condis',varargin)==1);
if isempty(optcd)
    condis = cell(1,py);
    if strcmp('pls',method)==1 || strcmp('pls2',method)==1 || strcmp('ppls',method)==1 || strcmp('cppls',method)==1 || strcmp('cpplsy',method)==1
        for i=1:py
            condis{1,i} = 'c';
        end
    else
        for i=1:py
            condis{1,i} = 'd';
        end
    end
else
    condis = varargin{optcd+1};
    if ~iscell(condis)
        condis = {condis};
    end
end
for i=1:py % Make 1,2,3,... classes from response
    if strcmp('d',condis{i})
        Y(:,i) = class123(Y(:,i));
    end
end

% gamma limits
optgam = find(strcmp('gamma',varargin)==1);
if isempty(optgam)
    lower = 0.5; upper = 0.5;
else
    lower = varargin{optgam+1}; upper = varargin{optgam+2};
end

% type of power
powerType = find(strcmp('powerType',varargin)==1);
if isempty(powerType)
    powerType = 0;
else
    powerType = varargin{powerType+1};
end

% Secondary response
optsec = find(strcmp('Ysec',varargin)==1);
if isempty(optsec)
    Ysec = [];
else
    Ysec = varargin{optsec+1};
end

% Discrimination method
optdisc = find(strcmp('disc',varargin)==1);
if isempty(optdisc)
    disc = 'lda';
else
    disc = varargin{optdisc+1};
end

% Transformation and retransformation
opttrans = find(strcmp('trans',varargin)==1);
if ~isempty(opttrans)
    transType1 = varargin{opttrans+1};
    transType2 = varargin{opttrans+2};
    if length(transType1) < py
        tT1 = cell(py,1);
        for i=1:py
            tT1{i} = transType1{1};
        end
        transType1 = tT1;
    end
    if length(transType2) < py
        tT2 = cell(py,1);
        for i=1:py
            tT2{i} = transType2{1};
        end
        transType2 = tT2;
    end

    for i=1:length(transType1)
        y = Y(:,i);
        Y(:,i) = eval(transType1{i});
    end
end

% Object weights
optwt = find(strcmp('wt',varargin)==1);
if isempty(optwt)
    wt = [];
else
    wt = varargin{optwt+1};
end

% Progress indicator
optprog = find(strcmp('progress',varargin)==1);
if isempty(optprog)
    progress = 1;
else
    progress = varargin{optprog+1};
end

% Posterior values
if sum(strcmp('post',varargin)) ~= 0
    more.post = cell(cvsegm,1);
else
    more = [];
end

% Loading weights
if sum(strcmp('loadW',varargin)) ~= 0
    more.loadW = cell(cvsegm,1);
end

% Prior weights for classification
optprior = find(strcmp('prior',varargin)==1);
if ~isempty(optprior)
    prior     = varargin{optprior+1};
    priorType = varargin{optprior+2};
    if priorType ~= 0 && ~isempty(optwt)
        error('Object weights specified both by "prior" and "wt"')
    end
else
    prior = cell(1,py);
    priorType = -1;
end


%% Construct primary response Yprim for CPPLS
d = find(strcmp('d',condis));
Yprim = [];
Ywidth = zeros(py,1);
for i=1:py
    if sum(i==d)>0
        Yd = dummy(Y(:,i));
        Yprim = [Yprim Yd];
        Ywidth(i) = size(Yd,2);
        if isempty(optprior) || isempty(prior{1,i})
            prior{1,i} = ones(1,Ywidth(i))./Ywidth(i);
        end
    else
        Ywidth(i) = 1;
        Yprim = [Yprim Y(:,i)];
    end
end


%% Construct object weights
if priorType > 0 % Object weights
    wt = wt_from_Pi(Yprim,prior{1});
end
if (priorType == 0 || priorType == 2)  && (py == 1) && isempty(prior{1}) % Group weights
    prior{1} = sum(Yprim)./ny;
end


%% Initializations
Ycv  = cell(py,1); % response after segmentation
Yhcv = cell(py,1); % cross-validated response
cur  = 1;          % placeholder for current segment
XNaN = isnan(X);   % check if X containts NaNs
if isempty(find(isnan(X)==1,1))
     XNaN = [];
end


%% Main cross-validation loop
if progress == 1
    fprintf('\nSegments computed\n 0%%');
end
if isfield(more,'post')
    more.post = cell(py,1);
end
if ~isempty(findstr(method, 'pp'))
    more.gam = zeros(ncomp,cvsegm);
end
for i = 1:cvsegm
    out = find(cv==i);       % test segment
    in  = setdiff(1:nx,out); % training segment
    % Training
    Xin  = X(in,:);
    Yin  = Y(in,:);         % response
    Ypin = Yprim(in,:);     % primary response
    Ysin = [];              % secondary response for cppls
    if isempty(wt) && priorType < 1 % Group weights only
        wtin = [];
    elseif priorType > 0 % Object weights
        wtin = wt_from_Pi(Ypin,prior{1});
    else
        wtin = wt(in,:)*nx/length(in);  % observation weights
    end
    if ~isempty(Ysec)
        Ysin = Ysec(in,:);
    end
    if ~isempty(XNaN)
        XNaNin  = XNaN(in,:);
        XNaNout = XNaN(out,:);
    end
    % Test
    Xout = X(out,:);
    lout = length(out);
    Yorder(out,1) = out; % Original order of data

    % Evaluate method
    if ~isempty(strfind(method, 'cppls')) && (strfind(method, 'cppls') == 1 || strfind(method, 'cpplsy') == 1)
        if ~isempty(wtin)
            [W, P, T, more.gam(:,i)] = feval(method, ncomp, Xin, Ypin, Ysin, lower, upper, [], wtin);
        else
            [W, P, T, more.gam(:,i)] = feval(method, ncomp, Xin, Ypin, Ysin, lower, upper);
        end
    elseif sum(method=='p') == 1 % pls, pls2 or plsda
        if sum(method=='a') == 1
            if priorType < 1
                [W, P, T] = feval(method, ncomp, Xin, Yin);
            else
                [W, P, T] = feval(method, ncomp, Xin, Yin, prior{1});
            end
        else
            [W, P, T] = feval(method, ncomp, Xin, Yin, wtin);
        end
    else % ppls or pplsda
        [W, P, T, more.gam(:,i)] = feval(method, ncomp, Xin, Yin, lower, upper, wtin);
    end
    
    if isfield(more,'loadW')
        more.loadW{i,1} = W;
    end

    for r=1:py
        Q = zeros(1,ncomp);
        B = zeros(px,ncomp);
        if strcmp('c',condis{r})
            mXin = mean(Xin);
            mYin = mean(Yin(:,r));
            for j=1:ncomp
                Q(:,j) = (Yin(:,r)'*T(:,j))/(T(:,j)'*T(:,j));
                B(:,j) =  W(:,1:j)/(P(:,1:j)'*W(:,1:j))*Q(:,1:j)';
            end
            Xoutc = Xout - ones(length(out),1)*mXin;
            Yout = ones(length(out),1)*mYin;
            Yhcv{r}(out,:) = [Yout (Xoutc*B)+Yout*ones(1,ncomp)];
            Ycv{r}(out,:) = Y(out,r);
        end
        if strcmp('d',condis{r})
            if isempty(XNaN)
                if strcmp(disc,'lda') % The standard classifcation method
                    if isfield(more,'post')
                        for j=1:ncomp
                            [Yhcv{r}(out,j),more.post{r}(out,:,j)] = lda(Xin*W(:,1:j),Yin(:,r),Xout*W(:,1:j),prior{r});
                        end
                    else
                        for j=1:ncomp
                            Yhcv{r}(out,j) = lda(Xin*W(:,1:j),Yin(:,r),Xout*W(:,1:j),prior{r});
                        end
                    end
                end
                if strcmp(disc,'qda')
                    if isfield(more,'post')
                        for j=1:ncomp
                            [Yhcv{r}(out,j),more.post{r}(out,:,j)] = qda(Xin*W(:,1:j),Yin(:,r),Xout*W(:,1:j),prior{r});
                        end
                    else
                        for j=1:ncomp
                            Yhcv{r}(out,j) = qda(Xin*W(:,1:j),Yin(:,r),Xout*W(:,1:j),prior{r});
                        end
                    end
                end
            else % X contains NaNs
                if strcmp(disc,'lda') % The standard classifcation method
                    if isfield(more,'post')
                        for j=1:ncomp
                            [Yhcv{r}(out,j),more.post{r}(out,:,j)] = lda(nanProd(Xin,W(:,1:j),XNaNin,X,XNaN),Yin(:,r),nanProd(Xout,W(:,1:j),XNaNout,X,XNaN),prior{r});
                        end
                    else
                        for j=1:ncomp
                            Yhcv{r}(out,j) = lda(nanProd(Xin,W(:,1:j),XNaNin,X,XNaN),Yin(:,r),nanProd(Xout,W(:,1:j),XNaNout,X,XNaN),prior{r});
                        end
                    end
                end
                if strcmp(disc,'qda')
                    if isfield(more,'post')
                        for j=1:ncomp
                            [Yhcv{r}(out,j),more.post{r}(out,:,j)] = qda(nanProd(Xin,W(:,1:j),XNaNin,X,XNaN),Yin(:,r),nanProd(Xout,W(:,1:j),XNaNout,X,XNaN),prior{r});
                        end
                    else
                        for j=1:ncomp
                            Yhcv{r}(out,j) = qda(nanProd(Xin,W(:,1:j),XNaNin,X,XNaN),Yin(:,r),nanProd(Xout,W(:,1:j),XNaNout,X,XNaN),prior{r});
                        end
                    end
                end
            end
            Ycv{r}(out,:) = Y(out,r);
        end
    end
    cur = cur + lout;  % Placeholder

    if progress == 1
        % Progress indication
        percent = round(100*i/cvsegm);
        last = round(100*(i-1)/cvsegm);
        if last <10
            fprintf('\b\b%d%%',percent)
        elseif last <100
            fprintf('\b\b\b%d%%',percent)
        else
            fprintf('\b\b\b\b%d%%',percent)
        end
    end
end

more.order = Yorder;

if ~isempty(opttrans)
    for i=1:py
%         y = Yhcv{i};
        Yhcv{i} = eval(transType2{i});
%         y = Ycv{i};
        Ycv{i} = eval(transType2{i});
    end
end


gfit = cell(r,1);
for r=1:py
    if strcmp('c',condis{r})
        for j=1:ncomp+1
            gfit{r}(j,1) = sqrt(mean((Yhcv{r}(:,j)-Ycv{r}).^2));
        end
    end
    if strcmp('d',condis{r})
        for j=1:ncomp
            gfit{r}(j,1) = (nx - sum(Yhcv{r}(:,j)==Ycv{r}))/nx;
        end
    end
end


%% Find a model with less components than the best
% which is not significantly worse
minis = zeros(py,4);
for i=1:py
    if strcmp('d',condis{i})
        cvsignificant = binocdf(nx*min(gfit{i}),nx*ones(1,ncomp),gfit{i}');
        pvalue = 0.05;
        indcv = find(cvsignificant > pvalue,1,'first');
        [mcv indmincv] = min(gfit{i});
        minis(i,:) = [indmincv, mcv, indcv, gfit{i}(indcv,1)];
    else
        cvsignificant = chi2cdf(((gfit{i}.^2)/(nx*min(gfit{i}.^2))).^(-1),nx);
        pvalue = 0.05;                                          %%%%%%% Significance level %%%%%%%
        indcv = find(cvsignificant > pvalue,1,'first')-1;
        [mcv indmincv] = min(gfit{i}(2:ncomp+1,1));
        minis(i,:) = [indmincv, mcv, indcv, gfit{i}(indcv+1,1)];
    end
end


%% Cross-validation segments
function [cv, s] = cvseg(n,s,type)
% [cv, s] = cvseg(n,s,type)
% Function for making crossvalidation segments
% n - number of samples
% s - number of segments
% type - type of crossvalidation
%		1='Full crossvalidation (leave one out)'
%		2='Random crossvalidation (samles are randomly picked for each segment)'
%		3='Systematic crossvalidation 111..-222..-333.. etc.
%		4='Systematic crossvalidation 123..-123..-123.. etc.
str=ceil(n/s);
if type==1 || s==n %full CV [123...n]
    cv=1:n;
    type=1;
    s = n;
end
if type == 2 %CV-segm randomly chosen
    cv=repmat(1:s,1,str);
    r=randperm(n);
    cv=cv(r);
end
if type==3 %CV-segm = [111 222 333 etc.]
    cv=repmat(1:s,1,str);
    cv=sort(cv(1:n));
end
if type==4 %%CV-segm = [123 123 123 etc.]
    cv=repmat(1:s,1,str);
    cv=cv(1:n);
end


%% Make 1,2,3,... classes out of numeric vector
function Xout = class123(Xinn)

n = size(Xinn,1);

coded = unique(Xinn);
n1 = size(coded,1);

Xout = zeros(n,1);
for i=1:n1
    Xout = Xout + i*(Xinn==coded(i));
end


%% Make dummy response
% function dum = dummy(X)
% 
% n = size(X,1);
% p = max(X);
% dum = zeros(n,p);
% for i=1:n
%     dum(i,X(i))=1;
% end
function dY = dummy(Y)
% Rutinen konverterer en vektor Y
% av K (uordnede) klasselabler 
% til en N*K indikatormatrise
Ys  = sort(Y(:));
dYs = [0;diff(Ys)];
s   = Ys(dYs~=0); % Tilsvarer sort(unique(Y))
m   = length(s);

dY = zeros(size(Y,1),m+1);
dY(:,1) = Y==Ys(1);
for i = 1:m
    dY(:,i+1) = Y==s(i);
end


%% Product (X*Y) where X contains NaNs
function result = nanProd(X,Y,XNaN,impX,impNaN)

if(nargin > 2)
    NaNs = find(XNaN==1);
    [n,p] = size(X);
    n1 = size(impX,1);
    u = zeros(n1,1);
    for i=1:length(NaNs)
        [a,b] = ind2sub([n,p],NaNs(i));
        for j=1:n1
            u(j) = nansum((impX(j,:)-X(a,:)).^2);
        end
        u(impNaN(:,b)==1) = Inf;
        [m1,m2] = min(u);
        X(NaNs(i)) = impX(m2,b);
    end
    result = X*Y;
    
else
    n1 = size(X,2);
    n2 = size(Y,2);
    X(XNaN==1) = 0;
    nNaNs = (n1./(n1-sum(XNaN,2)))*ones(1,n2);
    result = X*Y;
    result = result.*nNaNs;
end

%% Object weights
function wt = wt_from_Pi(dY,Pi)
% Calculation of object weights from Y-dummy
% and a vector of group priors
% 
% wt = wt_from_Pi(dY,Pi)

n = size(dY,1);

Pi0 = sum(dY)/n;    % Empirical priors
wt = dY*(Pi./Pi0)'; % Object weights