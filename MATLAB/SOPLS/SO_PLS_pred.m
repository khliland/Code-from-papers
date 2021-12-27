function [Ypred, YpredBlock, RMSEP, R2pred] = SO_PLS_pred(SO, Xt, Yt)

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
        if ~isempty(SO.V{i}(:,2:end))
            Xt{j} = Xt{j} - (Xt{i}*SO.V{i}(:,2:end))/(SO.T{i}(:,2:end)'*SO.T{i}(:,2:end))*SO.T{i}(:,2:end)' * X{j};
        end
    end
end

% Predict
YpredBlock = zeros(nt,size(SO.Q{1,1},1),SO.nb+1);
YpredBlock(:,:,1) = repmat(mean(SO.Y),nt,1);
for i = 1:SO.nb
    if ~isempty(SO.V{i})
        YpredBlock(:,:,i+1) = Xt{i}*SO.V{i}*SO.Q{i}';
    end
end
Ypred = sum(YpredBlock,3);

% Optionally compute RMSEP/R2pred
if exist('Yt','var') ~= 0 && nargout > 2
    SS = (Yt-Ypred).^2;
    RMSEP = sqrt(mean(SS));
    if nargout == 4
        R2pred = 1-sum(SS)./sum((Yt-ones(nt,1)*mean(SO.Y)).^2);
    end
end
