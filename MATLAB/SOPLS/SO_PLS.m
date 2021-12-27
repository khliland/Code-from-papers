function [SO] = SO_PLS(X,Y,ncomps, Yadd,lowers,uppers,weights,usePLS2)

% X       - cell (1xnb) containing all X blocks (ordered)
% Y       - matrix of responses
% ncomps  - number of PLS components to use in each block
% lowers  - (optional) lower bound for powers
% uppers  - (optional) upper bound for powers
% weights - (optional) object weights

nb    = size(X,2); % Number of X blocks

% Input checking
if exist('Yadd','var') == 0
    Yadd = [];
end
if exist('lowers','var') == 0 || isempty(lowers)
    lowers = ones(1,nb)*0.5;
end
if exist('uppers','var') == 0 || isempty(uppers)
    uppers = ones(1,nb)*0.5;
end
if length(lowers) == 1
    lowers = ones(1,nb)*lowers;
end
if length(uppers) == 1
    uppers = ones(1,nb)*uppers;
end
if exist('weights','var') == 0
    weights = [];
end
if exist('weights','var') == 0
    usePLS2 = false;
end

% Prepare data collection
W     = cell(1,nb);
P     = cell(1,nb);
Q     = cell(1,nb);
T     = cell(1,nb);
V     = cell(1,nb);
E     = cell(1,nb);
pow   = cell(1,nb);

% Use original values for a=1
Xorth = X;
for i=1:nb
    Xorth{i} = Xorth{i} - repmat(mean(Xorth{i}),size(Xorth{i},1),1);
end
E{1,1} = Y;

% Main loop
for a = 1:nb
%     for i = 1:a-1 % Orthogonalize on previous blocks
%         Xorth{1,a} = Xorth{1,a} - T{1,i}(:,2:end)/(T{1,i}(:,2:end)'*T{1,i}(:,2:end))*T{1,i}(:,2:end)' * Xorth{1,a};
%     end
    % Compute components
    if usePLS2
        [W{1,a}, P{1,a}, T{1,a},~,Q{1,a}] = pls2(ncomps(a), Xorth{1,a}, E{1,max(1,a-1)}); pow{1,a} = 0.5;
    else
        [W{1,a}, P{1,a}, T{1,a}, pow{1,a}, ~, ~, Q{1,a}] = cppls(ncomps(a), Xorth{1,a}, E{1,max(1,a-1)}, Yadd, lowers(a), uppers(a), weights);
    end
    E{1,a} = E{1,max(1,a-1)} - T{1,a}*Q{1,a}';
    V{1,a} = W{1,a}/(P{1,a}'*W{1,a});
    W{1,a} = [zeros(size(Xorth{1,a},2),1) W{1,a}];
    P{1,a} = [zeros(size(Xorth{1,a},2),1) P{1,a}];
    T{1,a} = [zeros(size(Xorth{1,a},1),1) T{1,a}];
    V{1,a} = [zeros(size(Xorth{1,a},2),1) V{1,a}];
    Q{1,a} = [zeros(size(Y,2),1) Q{1,a}];
end

% Return everything in one struct
SO.X = X; SO.Y = Y;
SO.Yadd    = Yadd;
SO.ncomps  = ncomps;
SO.lowers  = lowers;
SO.uppers  = uppers;
SO.weights = weights;
SO.nb      = nb;
SO.W = W;  SO.P = P;
SO.Q = Q;  SO.T = T;
SO.V = V;  SO.E = E;
SO.pow = pow;
