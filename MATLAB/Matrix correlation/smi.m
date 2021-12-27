function [SMI, Pval, signif, corrTU] = ...
    smi(X1,X2, ncomp1,ncomp2, B, alpha, replicates, T,U)
%% Similarity index for two matrices
% 
% INPUTS
% X1, X2:   matrices with corresponding objects
% ncompx:   number(s) of components
% progress: show progress (default = true)
% B:        number of bootstrap replicates
% alpha:    test level (default = 0.05)
% T,U:      orthonormal basises (default = svd(center(X)))
% 
% OUTPUTS
% SMI:    similarity index
% Pval :  P-values for H0: 1-SMI = 0
% signif: significance codes relating to Pval

% Missing or malformed arguments
if nargin < 5
    B = 2000;
end
if nargin < 6
    alpha = 0.05;
end

% Initialize
n      = size(X1,1);
SMI_B  = zeros(ncomp1, ncomp2, B);
Pval   = zeros(ncomp1, ncomp2);
signif = NaN(  ncomp1, ncomp2);
m      = bsxfun(@min,1:ncomp2,(1:ncomp1)');

% Replicates in the data must be handled together
if nargin < 7
    replicates = false;
end

% Orthonormal basis: Left singular vector(s)
if nargin < 9
    [T,~,~] = svds(center(X1),ncomp1);
    [U,~,~] = svds(center(X2),ncomp2);
end

% Component correlations
corrTU = corr(T,U);

% SMI
M   = cumsum(cumsum((T(:,1:ncomp1)'*U(:,1:ncomp2)).^2),2);
SMI = M./m;

% Permutations
if islogical(replicates)
    for b=1:B
        ind = randperm(n);
        M   = cumsum(cumsum((T(ind,1:ncomp1)'*U(:,1:ncomp2)).^2),2);
        SMI_B(:,:,b) = M./m;
    end
else
    segs  = replicates{1};
    nSeg  = max(segs);
    nRep  = replicates{2};
    for b=1:B
        indOut = randperm(nSeg);
        ind = zeros(n,1);
        for i=1:nSeg
            indIn  = randperm(nRep);
            ind(segs == indOut(i),1) = (i-1)*nRep+indIn;
        end
        M   = cumsum(cumsum((T(ind,1:ncomp1)'*U(:,1:ncomp2)).^2),2);
        SMI_B(:,:,b) = M./m;
    end
end

% Significance
for i=1:ncomp1
    for j=1:ncomp2
        Pval(i,j) = sum(max(SMI_B(i,j,:),1-SMI_B(i,j,:))<(SMI(i,j)))/B;
        if Pval(i,j) > alpha
            if i < j
                signif(i,j) = 1;
            elseif i > j
                signif(i,j) = -1;
            else
                signif(i,j) = 0;
            end
        end
    end
end
