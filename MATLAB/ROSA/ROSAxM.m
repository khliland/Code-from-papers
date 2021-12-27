function [beta, W, P, T, PtW, Wb, order, count, C, F] = ROSAxM(X, Y, A, combos)
% -------------------------------------------------------------------------
% ------------------ Ulf Indahl 15/03-2014 --------------------------------
% -------------------------------------------------------------------------
% Multiresponsutgave
%
n = size(X{1},1);
nb = length(X);
ny = size(Y,2);
m  = zeros(nb,1);
for i=1:nb
    m(i) = size(X{i},2);
    X{i} = center(X{i});
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
r = zeros(n,ny,nb);
C = zeros(nb,A);
F = zeros(nb,A);
if combos > 0
    comb = unique(all_comb_lim([], [], nb,combos,1),'rows'); % All block combinations as a matrix
%     comb = unique(comb(:,1:combos),'rows');
end

% -------------------- Solution of the MBPLS-problem -----------------------
for a = 1:A
    for i=1:nb
        [ww,ss,~] = svds(X{i}'*Y,1);
        w{i} = ww.*ss;
        t(:,i) = X{i}*w{i};
    end
    
    if a > 1
        for i=1:nb
            t(:,i) = t(:,i) - T(:,1:a-1)*(T(:,1:a-1)'*t(:,i));
        end
    end
    for i=1:nb
        t(:,i) = t(:,i)/norm(t(:,i)); y1hat = t(:,i)*(t(:,i)'*Y); r(:,:,i) = Y-y1hat;
    end
    
    r(:,:,count'>=m) = Inf;
    if all(r==Inf)
        error(['Too many components, max=' num2str(min(sum(m),n-1))])
    end
    [mn,mr] = min(squeeze(sum(sum(r.^2,2),1))); % Minimum norm
    C(:,a) = corr(t(:,mr),t); C(mr,a) = 1; % Correlation between block scores.
    F(:,a) = sqrt(squeeze(mean(mean(r.^2,2),1)))'; % Block-wise fit to residual response.
    
    rrr = mn+1;
    if nargin > 3 && combos > 0 % Check all block combinations
        wwi = cell2mat(w);
        rrr  = zeros(size(comb,1),1);
        for j = 1:size(comb,1) % Loop over block combinations
            nind = 1:nb; nind(comb(j,:)) = []; % Prepare for zeroing out other block contributions to w
            ww = wwi;
            for i = 1:length(nind)
                ww(sum(m(1:(nind(i)-1)))+(1:m(nind(i)))) = 0; % Common loading weighs for common component
            end
            tt = XX*ww;
            tt = tt - T(:,1:a-1)*(T(:,1:a-1)'*tt);
            tt = tt./norm(tt); y1hat = tt*(tt'*Y); rr = Y-y1hat;
            rrr(j,1) = sum(rr.^2);
        end
    end
    
    if min(rrr) < mn % Multi block candidate scores
        [~,rri] = min(rrr);
        nind = 1:nb; nind(comb(rri,:)) = [];
        ww = wwi;
        for i = 1:length(nind)
            ww(sum(m(1:(nind(i)-1)))+(1:m(nind(i)))) = 0; % Common loading weighs for common component
        end
        tt = XX*ww;
        tt = tt - T(:,1:a-1)*(T(:,1:a-1)'*tt);
        tt = tt./norm(tt); y1hat = tt*(tt'*Y); rr = Y-y1hat;
        Y = rr; order{a} = unique(comb(rri,:));
        if count(end) > 0
            ww = ww - Wb{end}(:,1:count(end))*(Wb{end}(:,1:count(end))'*ww);
        end
        count(end) = count(end)+1;
        Wb{end}(:,count(end)) = ww./norm(ww);
        T(:,a) = tt;
        W(:,a) = Wb{end}(:,count(end));
    
    else % Single block candidate scores
        Y = r(:,:,mr); order{a} = mr;
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
beta = zeros(size(XX,2),A,ny);
for i=1:ny
    beta(:,:,i) = cumsum(bsxfun(@times,R, q(i,:)),2);   % The X-regression coefficients
end
beta = permute(beta,[1,3,2]);

for i=1:nb+1
    Wb{i} = Wb{i}(:,1:count(i));
end
%Yhat = T*(T'*y);

%% All combinations of blocks as a matrix (found recursively)
% function comb = all_comb(comb, vec, n, pos)
% if pos > n
%     comb = [comb; sort(vec)];
% else
%     for i=1:n
%         vec(1,pos) = i;
%         comb = all_comb(comb, vec, n, pos+1);
%     end
% end

function comb = all_comb_lim(comb, vec, nb, n, pos)
if pos > n
    comb = [comb; sort(vec)];
else
    for i=1:nb
        vec(1,pos) = i;
        comb = all_comb_lim(comb, vec, nb, n, pos+1);
    end
end