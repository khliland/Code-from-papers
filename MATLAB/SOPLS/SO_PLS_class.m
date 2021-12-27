function [Ypred, YpredBlock, RMSEP, R2pred] = SO_PLS_class(SO, Xt, Yt)

X  = SO.X;
nt = size(Xt{1},1);
n  = size(X{1},1);

% Center X and Xt using SO.X
for i=1:SO.nb
    Xt{i} = Xt{i}-ones(nt,1)*mean(X{i});
    X{i}  = X{i} -ones(n,1)*mean(X{i});
end
% Orhtogonalize all remaining blocks with respect to current
for i=1:SO.nb
    for j=(i+1):SO.nb
        Xt{j} = Xt{j} - (Xt{i}*SO.V{i}(:,2:end))/(SO.T{i}(:,2:end)'*SO.T{i}(:,2:end))*SO.T{i}(:,2:end)' * X{j};
    end
end

ncomps = zeros(SO.nb,1);
for i=1:SO.nb
    ncomps(i,1) = size(SO.V{i},2);
end

if sum(ncomps) == SO.nb
    [~,m] = max(sum(SO.Y));
    Ypred = ones(nt,1)*m;
else
    % Classify
    T  = zeros(n,sum(ncomps));
    Tt = zeros(nt,sum(ncomps));
    Y  = SO.Y*(1:size(SO.Y,2))';
    for i = 1:SO.nb
        T(:,sum(ncomps(1:(i-1)))+(1:ncomps(i)))  = SO.T{i};
        Tt(:,sum(ncomps(1:(i-1)))+(1:ncomps(i))) = Xt{i}*SO.V{i};
    end
    T(:,[1 cumsum(ncomps(1:end-1)')+1]) = [];
    Tt(:,[1 cumsum(ncomps(1:end-1)')+1]) = [];
    Ypred = lda(T,Y,Tt);
end
% Ypred = sum(YpredBlock,3);

% Optionally compute RMSEP/R2pred
if exist('Yt','var') ~= 0 && nargout > 2
    SS = (Yt-Ypred).^2;
    RMSEP = sqrt(mean(SS));
    if nargout == 4
        R2pred = 1-sum(SS)./sum((Yt-ones(nt,1)*mean(SO.Y)).^2);
    end
end
