function [Yhcv, RMSECV] = crossval_ROSAx(X,Y, ncomp, order, nseg, cvtype)

n  = size(X{1},1);
nb = length(X);
if cvtype~=5
    [cv, nseg] = cvseg(n,nseg,cvtype);
else
    cv = nseg;
    nseg = max(cv);
end

Yhcv = zeros(n,ncomp+1);

for i=1:nseg
    Xin  = cell(1,nb);
    Xout = cell(1,nb);
    for a=1:nb
        Xin{a}  = X{a}(cv~=i,:);
        Xout{a} = X{a}(cv==i,:);
    end
    Yin  = Y(cv~=i,:);
    Yout = Y(cv==i,:);
    
    beta = ROSAx(Xin, Yin, ncomp, order); % FIXME: combos byttet med order
    Yhcv(cv==i,2:end) = predict_ROSAx(beta, Xin, Yin, Xout, Yout);
    Yhcv(cv==i,1) = mean(Yin);
end

RMSECV = sqrt(mean(bsxfun(@minus, Yhcv, Y).^2))';


%% 
function [beta, W, P, T, PtW, Wb, order, count] = ROSAx(X, Y, A, order)
% -------------------------------------------------------------------------
% ------------------ Ulf Indahl 15/03-2014 --------------------------------
% -------------------------------------------------------------------------
n = size(X{1},1);
nb = length(X);
m  = zeros(nb,1);
for i=1:nb
    m(i) = size(X{i},2);
    X{i} = center(X{i});
end
XX = cell2mat(X);
m(nb+1) = sum(m);
% order = cell(1,A);  % Keeps track of active blocks order
count = zeros(1,nb+1); % Counts each time a block is active
T  = zeros(n,A);     % Orthonormal scores
Wb = cell(nb+1,1);
for i=1:nb+1 % Orthonormal weights of blocks
    Wb{i} = zeros(m(i),A);
end
W = zeros(m(end),A);
Y = center(Y);
y = Y;
w = cell(nb,1);
t = zeros(n,nb);
r = zeros(n,nb);

% -------------------- Solution of the MBPLS-problem -----------------------
for a = 1:A
    for i=1:nb
        w{i} = X{i}'*Y;
        t(:,i) = X{i}*w{i};
    end
    
    if a > 1
        for i=1:nb
            t(:,i) = t(:,i) - T(:,1:a-1)*(T(:,1:a-1)'*t(:,i));
        end
    end
    for i=1:nb
        t(:,i) = t(:,i)/norm(t(:,i)); y1hat = t(:,i)*(t(:,i)'*Y); r(:,i) = Y-y1hat;
    end
    
    if length(order{a})>1 % Two or more candidate scores are highly correlated to the response
        cr = order{a};
        nind = 1:nb; nind(cr) = [];
        ww = cell2mat(w);
        for i = 1:length(nind)
            ww(sum(m(1:(nind(i)-1)))+(1:m(nind(i)))) = 0; % Common loading weighs for common component
        end
        tt = XX*ww;
        tt = tt - T(:,1:a-1)*(T(:,1:a-1)'*tt);
        tt = tt./norm(tt); y1hat = tt*(tt'*Y); rr = Y-y1hat;
        Y = rr; order{a} = unique(cr);
        if count(end) > 0
            ww = ww - Wb{end}(:,1:count(end))*(Wb{end}(:,1:count(end))'*ww);
        end
        count(end) = count(end)+1;
        Wb{end}(:,count(end)) = ww./norm(ww);
        T(:,a) = tt;
        W(:,a) = Wb{end}(:,count(end));
    
    else % No candiate scores are highly correlated to the response
        mr = order{a};
        Y = r(:,mr); order{a} = mr;
        if count(mr) > 0
            w{mr} = w{mr} - Wb{mr}(:,1:count(mr))*(Wb{mr}(:,1:count(mr))'*w{mr});
        end
        count(mr) = count(mr)+1;
        Wb{mr}(:,count(mr)) = w{mr}./norm(w{mr});
        T(:,a) = t(:,mr);
        W(sum(m(1:(mr-1)))+(1:m(mr)),a) = Wb{mr}(:,count(mr));
    end
end
% -------------------------------------------------------------------------
% ---- Postprocessing to find regression coeffs & other key matrices -----
P    = cell2mat(X)'*T; % X-loadings.
PtW  = triu(P'*W); % The W-coordinates of (the projected) P.
R    = W/PtW;      % The "SIMPLS weights".
q    = y'*T;       % Regression coeffs (Y-loadings) for the orthogonal scores
beta = cumsum(bsxfun(@times,R, q),2);   % The X-regression coefficients

for i=1:nb+1
    Wb{i} = Wb{i}(:,1:count(i));
end
%Yhat = T*(T'*y);


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

%% Prediction
function [Ypred, RMSEP] = predict_ROSAx(beta, X,Y, Xt, Yt)

ncomp = size(beta,2);
n     = size(Xt{1},1);

% Centering and preparation
Xt = bsxfun(@minus, cell2mat(Xt), mean(cell2mat(X)));
Ypred = zeros(n,ncomp);

% Prediction
for i=1:ncomp
    Ypred(:,i) = Xt*beta(:,i) + mean(Y);
end

% Error calculation
if nargin > 4
    RMSEP = sqrt(mean(bsxfun(@minus, Ypred, Yt).^2))';
end
