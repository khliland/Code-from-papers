function [beta, W, P, T, PtW, Wb, order, count, BC] = ROSA(X, Y, A, cTol)
% -------------------------------------------------------------------------
% ------------------ Ulf Indahl 15/03-2014 --------------------------------
% -------------------------------------------------------------------------
n = size(X{1},1);
nb = length(X);
m  = zeros(nb,1);
for i=1:nb
    m(i) = size(X{i},2);
end
XX = cell2mat(X);
m(nb+1) = sum(m);
order = cell(1,A);  % Keeps track of active blocks order
count = zeros(1,nb+1); % Counts each time a block is active
T  = zeros(n,A);     % Orthonormal scores
Wb = cell(nb+1,1);
for i=1:nb+1 % Orthonormal weights of blocks
    Wb{i} = zeros(m(i),A);
end
W = zeros(m(end),A);
y = Y;
w = cell(nb,1);
t = zeros(n,nb);
r = zeros(n,nb);
BC  = zeros(nb,nb,A);  % Block score correlations
BCp = zeros(nb,nb,A);  % Block score correlation p-values
% -------------------- Solution of the MBPLS-problem -----------------------
for a = 1:A
    for i=1:nb
        w{i} = X{i}'*Y;
        t(:,i) = X{i}*w{i};
    end
    
    if a > 1
        for i=1:nb % Replace this loop with: t = t - T(:,1:a-1)*(T(:,1:a-1)'*t);
            t(:,i) = t(:,i) - T(:,1:a-1)*(T(:,1:a-1)'*t(:,i));
        end
    end
    for i=1:nb
        t(:,i) = t(:,i)/norm(t(:,i)); 
        y1hat = t(:,i)*(t(:,i)'*Y); r(:,i) = Y-y1hat; % Replace with r(:,i) = Y - t(:,i)*(t(:,i)'*Y);
    end
    
    [cc,pp] = corr(t);
    BC(:,:,a)  = triu(cc,1);
    BCp(:,:,a) = triu(pp,1);
    ind = [];
    [mn,mr] = min(sum(r.^2)); % Minimum norm

    if nargin > 3 && cTol > 0
        ind = find_common(abs(BCp(:,:,a)), cTol, nb); % Find groups of score candidates with high correlation to each other.
        ww = cell2mat(w);
        nind = setdiff(1:nb,ind);
        for i = 1:length(nind)
            ww(sum(m(1:(nind(i)-1)))+(1:m(nind(i)))) = 0; % Common loading weighs for common component
        end
        tt = XX*ww;
        tt = tt - T(:,1:a-1)*(T(:,1:a-1)'*tt);
        tt = tt./norm(tt); y1hat = tt*(tt'*Y); rr = Y-y1hat;
    end
    
    if ~isempty(ind) && sum(rr.^2) < mn % Multi block candidate scores
        Y = rr; order{a} = ind;
        if count(end) > 0
            ww = ww - Wb{end}(:,1:count(end))*(Wb{end}(:,1:count(end))'*ww);
        end
        count(end) = count(end)+1;
        Wb{end}(:,count(end)) = ww./norm(ww);
        T(:,a) = tt;
        W(:,a) = Wb{end}(:,count(end));
    
    else % Single block candidate scores
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

function ind = find_common(C, cTol, nb)
n = size(C,1);
C = C + tril(ones(n,n));
c = find(C<cTol);
[i1,i2] = ind2sub([nb,nb],c);
if ~isempty(i1)
    ind = unique([i1,i2]); % Bør oppdateres for å ta hensyn til grupperinger
else
    ind = [];
end

function comb = all_comb(comb, vec, n, pos)
if pos > n
    comb = [comb; sort(vec)];
else
    for i=1:n
        vec(1,pos) = i;
        comb = all_comb(comb, vec, n, pos+1);
    end
end