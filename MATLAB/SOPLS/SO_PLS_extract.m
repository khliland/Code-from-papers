%% Extract elements from SO_PLS object
function X = SO_PLS_extract(SO, type, block, ncomps)
if exist('ncomps','var')==0
    ncomps = SO.ncomps;
end
if block < length(ncomps) % Compensate for memory saving
    ncomps(block+1:end) = ones(1,length(ncomps(block+1:end)));
end

X = [];

% X scores
if strcmp(type, 'T') || strcmp(type, 'scores')
    X = cellget(SO.T, ncomps,block);
end
% X loadings
if strcmp(type, 'P') || strcmp(type, 'loadings')
    X = cellget(SO.P, ncomps,block);
end
% X loadings
if strcmp(type, 'PX') || strcmp(type, 'loadings')
    X = SO.X{block};
    T = cellget(SO.T, ncomps,block); T = T(:,2:ncomps(block));
    X = [zeros(size(X,2),1),(X-mean(X))'*T./sum(T.^2)];
end
% X loading weights
if strcmp(type, 'W') || strcmp(type, 'loadingweights')
    X = cellget(SO.W, ncomps,block);
end
% X projections
if strcmp(type, 'V') || strcmp(type, 'projections')
    X = cellget(SO.V, ncomps,block);
end
% Y loadings
if strcmp(type, 'Q') || strcmp(type, 'yloadings')
    X = cellget(SO.Q, ncomps,block);
end
% Regression coefficients
if strcmp(type, 'B') || strcmp(type, 'coefficients')
    X1 = cellget(SO.V, ncomps,block);
    X2 = cellget(SO.Q, ncomps,block);
    X = X1(:,1);
    if ncomps(block) > 1
        X = [X X1(:,2:end)*X2(:,2:end)'];
    end
end

% if ~isempty(X)
%     X = X(:,2:end);
% end


%% Get cell using vector
function mat = cellget(X, dims,block)
expr = 'mat = X(';
for i=1:length(dims)
    if i==block
        expr = [expr '1:'];
    end
    expr = [expr num2str(dims(i)) ','];
end
expr = [expr num2str(block) ');'];
eval(expr);
mat = cell2mat(mat(:)');
