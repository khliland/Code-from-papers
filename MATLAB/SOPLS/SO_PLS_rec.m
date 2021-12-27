function [SO] = SO_PLS_rec(X,Y,ncomps, restricted, Yadd,DynYadd, lowers,uppers, weights, progress, conf)

% X       - cell (1xnb) containing all X blocks (ordered)
% Y       - matrix of responses
% ncomps  - number of PLS components to use in each block
% restricted - true/false 1/0 restrict model to columnspace of X{b}
% Yadd    - matrix of additional responses (reused by all blocks)
% DynYadd - cell of additional responses per block
%           (typically block scores from all other blocks,
%            ex. DynAdd{2:nblock} is used when modelling block 1)
% lowers  - (optional) lower bound(s) for powers
% uppers  - (optional) upper bound(s) for powers
% weights - (optional) object weights
% progress- show progress (1)?
% conf    - confidence interval of Lenth's method

nb    = size(X,2); % Number of X blocks

% Input checking
if exist('restricted','var') == 0
    restricted = false;
end
if exist('Yadd','var') == 0
    Yadd = [];
end
if exist('DynYadd','var') == 0
    DynYadd = [];
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
if exist('progress','var') == 0
    progress = 0;
end
if exist('conf','var') == 0
    conf = 0;
end

% Prepare for data collection
ncomps = ncomps + 1; % Includes zero component
SO.X = X; SO.Y = Y;
SO.restricted = restricted;
SO.Yadd   = Yadd;
SO.DynYadd= DynYadd;
if length(ncomps) == 1
    SO.maxcomp = ncomps;
    ncomps = repmat(ncomps,1,nb);
else
    SO.maxcomp = [];
end
if length(conf) == 1
    conf = repmat(conf,nb,1);
end
SO.ncomps = ncomps;
SO.lowers = lowers;
SO.uppers = uppers;
SO.weights= weights;
SO.nb     = nb;
SO.W      = cell([ncomps nb]); % Bytt til nb lang celle bestående av fine underdimensjoner
SO.P      = cell([ncomps nb]);
SO.Q      = cell([ncomps nb]);
SO.T      = cell([ncomps nb]);
SO.V      = cell([ncomps nb]);
SO.E      = cell([ncomps nb]);
SO.pow    = cell([ncomps nb]);
SO.ssqx   = cell([ncomps nb]);
SO.ssqy   = cell([ncomps nb]);
SO.conf   = conf;

Xc = X;
for i=1:nb
    Xc{i} = Xc{i} - repmat(mean(Xc{i}),size(Xc{i},1),1);
end

% Prepare progress printing
if progress == 1
    fprintf('Current component: ',progress)
    print_progress(zeros(1,nb),ncomps,0)
end

% Make struct global and run recursive algorithm
global SOglob
SOglob = SO;
SO_loop([],Y,Xc,progress,[]);
SO = SOglob;
clear SOglob;
SO.ncomps = SO.ncomps-1;

%% Recursive loop function
function [] = SO_loop(comp,E,Xorth_in,progress,Tb)
global SOglob
nb = SOglob.nb;      % Number of blocks
cb = length(comp)+1; % Current block


if cb <= nb % Check if recursion has reached lowest level
%     [W,P,T,~,~,~,Q, ssqx, ssqy] = pls2(SOglob.ncomps(cb),Xorth_in{cb},E);
    Yadd = SOglob.Yadd;
    if ~isempty(SOglob.DynYadd)
        Yadd = [Yadd cell2mat(SOglob.DynYadd(setdiff(1:nb,cb)))];
    end

    % Display progress
    if progress==1
        print_progress(comp-1,SOglob.ncomps,cb)
    end
            
%     [W, P, T, pow, ~, ~, Q, ~, ~, ~, ssqx, ssqy] = cppls(SOglob.ncomps(cb)-1, Xorth_in{cb}, E, Yadd, SOglob.lowers(cb), SOglob.uppers(cb), SOglob.weights, SOglob.conf(cb));
    restricted = SOglob.restricted;
    if restricted && ~isempty(Tb)
        [W, P, T, ~, Q, ~,~, ssqx, ssqy] = pls2restricted(SOglob.ncomps(cb)-1, Xorth_in{cb}, E, Tb); pow = repmat(0.5,SOglob.ncomps(cb)-1,1);
    else
        [W, P, T, ~, Q, ~,~, ssqx, ssqy] = pls2(SOglob.ncomps(cb)-1, Xorth_in{cb}, E); pow = repmat(0.5,SOglob.ncomps(cb)-1,1);
    end
    V = W/(P'*W);

    % Zero component solution
    Xorth = Xorth_in;

    n = size(T,1);
    p = size(W,1);
    ny = size(Q,1);
    comps = [comp 1];
    
    % Housekeeping / storage
    c = ones(1,nb); c(1:length(comps)) = comps; % Place holder
    SOglob.W    = vec_ind(SOglob.W,   '{',c,'}',zeros(p,1), cb); % Concatenate at correct place
    SOglob.P    = vec_ind(SOglob.P,   '{',c,'}',zeros(p,1), cb); % Concatenate at correct place
    SOglob.T    = vec_ind(SOglob.T,   '{',c,'}',zeros(n,1), cb); % Concatenate at correct place
    SOglob.Q    = vec_ind(SOglob.Q,   '{',c,'}',zeros(ny,1),cb); % Concatenate at correct place
    SOglob.pow  = vec_ind(SOglob.pow, '{',c,'}',0.5,        cb); % Concatenate at correct place
    SOglob.V    = vec_ind(SOglob.V,   '{',c,'}',zeros(p,1), cb); % Concatenate at correct place
    SOglob.E    = vec_ind(SOglob.E,   '{',c,'}',zeros(n,ny),cb); % Concatenate at correct place
    SOglob.ssqx = vec_ind(SOglob.ssqx,'{',c,'}',0,          cb); % Concatenate at correct place
    SOglob.ssqy = vec_ind(SOglob.ssqy,'{',c,'}',0,          cb); % Concatenate at correct place

    % Process next block based on current path through components and blocks
    SO_loop(comps,E,Xorth,progress,Tb);
    if ~isempty(SOglob.maxcomp)
        maxa = min(SOglob.maxcomp-sum(comps-1)-1,SOglob.ncomps(cb)-1);
    else
        maxa = SOglob.ncomps(cb)-1;
    end
    % 1+component solutions
    for a = 1:maxa
        Xorth = Xorth_in;
        for j=(cb+1):nb % Orhtogonalize all remaining blocks with respect to previous
            Xorth{1,j} = Xorth{1,j} - T(:,1:a)/(T(:,1:a)'*T(:,1:a))*T(:,1:a)' * Xorth{1,j};
        end
        Ec = E - T(:,1:a)*Q(:,1:a)'; % Error after current component/block
        % FIXME: Er det egentlig så lurt å deflatere når en eventuelt har kategorier i respons?
        % NEI!
%         Ec = E;
        
        comps = [comp 1+a];

        % Housekeeping / storage
        c = ones(1,nb); c(1:length(comps)) = comps; % Place holder
        SOglob.W    = vec_ind(SOglob.W,   '{',c,'}',W(:,a), cb); % Concatenate at correct place
        SOglob.P    = vec_ind(SOglob.P,   '{',c,'}',P(:,a), cb); % Concatenate at correct place
        SOglob.T    = vec_ind(SOglob.T,   '{',c,'}',T(:,a), cb); % Concatenate at correct place
        SOglob.Q    = vec_ind(SOglob.Q,   '{',c,'}',Q(:,a), cb); % Concatenate at correct place
        SOglob.pow  = vec_ind(SOglob.pow, '{',c,'}',pow(a), cb); % Concatenate at correct place
        SOglob.V    = vec_ind(SOglob.V,   '{',c,'}',V(:,a), cb); % Concatenate at correct place
        SOglob.E    = vec_ind(SOglob.E,   '{',c,'}',Ec,     cb); % Concatenate at correct place
        SOglob.ssqx = vec_ind(SOglob.ssqx,'{',c,'}',ssqx(a),cb); % Concatenate at correct place
        SOglob.ssqy = vec_ind(SOglob.ssqy,'{',c,'}',ssqy(a),cb); % Concatenate at correct place
        
        % Process next block based on current path through components and blocks
        SO_loop(comps,Ec,Xorth,progress,[Tb,T(:,1:a)]);
    end
end


%% Fill all cells from current and out (for øyeblikket ubrukt)
% function X = concat_from(X, vec, ncomps, comps, nb, cb)
% c = ones(1,nb); c(1:length(comps)) = comps; % Place holder
% i=1; % Iterator
% left = elem_left(comps,ncomps,cb); % Size of remaining portion
% while i <= left
%     X = vec_ind(X,'{',c,'}',vec, cb); % Concatenate at correct place
%     c(end) = c(end)+1; % Move to next
%     i = i+1;           % -----||-----
%     if i < left
%         for j=nb:-1:1      % -----||-----
%             if c(j) > ncomps(j)
%                 c(j) = 1;
%                 c(j-1) = c(j-1)+1;
%             end
%         end
%     end
% end


%% Insert using vector
function X = vec_ind(X,par1,vec,par2,ins, cb) %#ok<INUSL>
expr = ['X' par1];
for i=1:length(vec)
    expr = [expr num2str(vec(i)) ',']; %#ok<AGROW>
end
expr = [expr num2str(cb) par2];
expr = [expr '=[' expr ' ins];'];
eval(expr);


%% Number of elements left
function num = elem_left(comps, ncomps,cb)
left = [comps ones(1,size(ncomps,2)-size(comps,2))];
left = ncomps - left;
for i=1:(size(ncomps,2)-1)
    left(1:(end-i)) = left(1:(end-i)) .* ncomps((end-i+1));
end
num = sum(left(cb:end))+1;

%% Print progress
function [] = print_progress(comps,ncomps,cb)
long = ncomps>=10;
lc = length(comps);
ln = length(ncomps);
if lc<ln
    comps = [comps zeros(1,ln-lc)];
end
out = '';
for i=1:length(ncomps)
    if long(i) && comps(i)<10
        out = [out ' '];
    end
    if i<=cb
        out = [out num2str(comps(i)) ', '];
    else
        out = [out 'x, '];
    end
end
if cb<ln
    out = [out(1:(end-3)) 'x'];
else
    out = [out(1:(end-3)) '?'];
end
if cb==0
    fprintf(out,0)
elseif sum(comps) == sum(ncomps-1)
    fprintf(repmat('\b',1,length(out)+19),0)
else
    fprintf([repmat('\b',1,length(out)) out],0)
end
