function [SOs boot ssqy] = SO_PLS_bootstrap(SO, B)

% Prepare for bootstrapping
N         = size(SO.Y,1);
Ycal      = SO_PLS_pred(SO,SO.X,SO.Y);
residuals = SO.Y - Ycal;
SOs       = cell(B,1);
X         = SO.X;
ncomps    = SO.ncomps;

% Main bootstrap loop
for i=1:B
    Y_boot = Ycal + residuals(unidrnd(N,N,1),:);
    SOs{i,1} = SO_PLS_fixed(X,Y_boot,SO.ncomps,SO.Yadd,SO.pow,SO.weights);
end
SOall = SO_PLS_fixed(X,SO.Y,SO.ncomps,SO.Yadd,SO.pow,SO.weights);
ssqy  = SOall.ssqy;

% Concatenate results
nb = length(SO.W);
boot = SO;
boot = rmfield(boot, 'V');
boot.ssqy = cell(1,nb);
for i=1:B
    for j=1:nb
        for k = 1:ncomps(j)
            if corr(SOs{i}.W{j}(:,k), SO.W{j}(:,k+1)) < 0
                SOs{i}.W{j}(:,k) = -SOs{i}.W{j}(:,k);
            end
            if corr(SOs{i}.P{j}(:,k), SO.P{j}(:,k+1)) < 0
                SOs{i}.P{j}(:,k) = -SOs{i}.P{j}(:,k);
            end
            if corr(SOs{i}.Q{j}(:,k), SO.Q{j}(:,k+1)) < 0
                SOs{i}.Q{j}(:,k) = -SOs{i}.Q{j}(:,k);
            end
            if corr(SOs{i}.T{j}(:,k), SO.T{j}(:,k+1)) < 0
                SOs{i}.T{j}(:,k) = -SOs{i}.T{j}(:,k);
            end
        end
        if i == 1
            boot.W{j} = SOs{i}.W{j};
            boot.P{j} = SOs{i}.P{j};
            boot.Q{j} = SOs{i}.Q{j};
            boot.T{j} = SOs{i}.T{j};
            boot.E{j} = SOs{i}.E{j};
            boot.ssqy{j} = SOs{i}.ssqy{j};
        else
            boot.W{j}(:,:,i) = SOs{i}.W{j};
            boot.P{j}(:,:,i) = SOs{i}.P{j};
            boot.Q{j}(:,:,i) = SOs{i}.Q{j};
            boot.T{j}(:,:,i) = SOs{i}.T{j};
            boot.E{j}(:,:,i) = SOs{i}.E{j};
            boot.ssqy{j}(:,i) = SOs{i}.ssqy{j};
        end
    end
end


%% SO-PLS using CPPLS with fixed gammas
function [SO] = SO_PLS_fixed(X,Y,ncomps, Yadd,gamma,weights)

% X       - cell (1xnb) containing all X blocks (ordered)
% Y       - matrix of responses
% ncomps  - number of PLS components to use in each block
% gamma   - gammas for power in cppls
% weights - (optional) object weights

nb    = size(X,2); % Number of X blocks

% Prepare data collection
W     = cell(1,nb);
P     = cell(1,nb);
Q     = cell(1,nb);
T     = cell(1,nb);
V     = cell(1,nb);
E     = cell(1,nb);
pow   = cell(1,nb);
ssqy  = cell(1,nb);

% Use original values for a=1
Xorth  = X;
E{1,1} = Y;

% Main loop
for a = 1:nb
    for i = 1:a-1 % Orthogonalize on previous blocks
        Xorth{1,a} = Xorth{1,a} - T{1,i}/(T{1,i}'*T{1,i})*T{1,i}' * Xorth{1,a};
    end
    % Compute components
    [W{1,a}, P{1,a}, T{1,a}, pow{1,a}, ~, ~, Q{1,a}, ~, ~, ~, ~, ssqy{1,a}] = cppls_fixed(ncomps(a), Xorth{1,a}, E{1,max(1,a-1)}, Yadd, gamma{a}, weights);
    E{1,a} = Y - T{1,a}*Q{1,a}';
    V{1,a} = W{1,a}/(P{1,a}'*W{1,a});
end

% Return everything in one struct
SO.X = X; SO.Y = Y;
SO.Yadd    = Yadd;
SO.ncomps  = ncomps;
SO.gamma   = gamma;
SO.weights = weights;
SO.nb      = nb;
SO.W = W;  SO.P = P;
SO.Q = Q;  SO.T = T;
SO.V = V;  SO.E = E;
SO.pow  = pow;
SO.ssqy = ssqy;