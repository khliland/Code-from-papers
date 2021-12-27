function [group,d,dist] = qda(X,Y,varargin)
% Quadratic discriminant analysis
% 
% ---- Inputs ----
% X = predictors
% Y = groups (1,2,..,G)
% 
% ---- Optional ----
% Xnew  = validation predictors
% Prior = group prior probabilities
% 
% ---- Outputs ---
% group = classified group
% d     = posterior probabilities
% dist  = distance from group centers
% 

[Yd,m] = dummy(Y);  % Dummy response and max(Y)

% Missing input variables
if isempty(varargin)
    Xnew = X;
    Prior = ones(m,1)./m;
elseif length(varargin) == 1
    Xnew = varargin{1};
    Prior = ones(m,1)./m;
else
    Xnew  = varargin{1};
    if isempty(varargin{2})
        Prior = ones(m,1)./m;
    else
        Prior = varargin{2}';
    end
end

% Dimensions
p  = size(X,2);
n2 = size(Xnew,1);

% Declaration of variables
Sa = zeros(1,m);     % Determinants
Si = zeros(m*p,m*p); % Block diagonal matrix containing all Si

N = sum(Yd);              % Group sizes
Xs = (Yd'*Yd)\Yd'*X;      % Mean for each group
Xc = X-Xs(Y,:);           % Centering

% Covariance matrices
for i=1:m
    xi = Xc(Y==i,:);
    S = xi'*xi./(N(i)-1);
    Si((i-1)*p+(1:p),(i-1)*p+(1:p)) = inv(S); % 'inv' is faster, 'pinv' more robust
    Sa(1,i) = det(S);
end

% Common ellements
Xst = Xs';                                       % Transposed means
R = repmatS(Xnew',[m,1])-repmatS(Xst(:),[1,n2]); % Center using all groups
Q = R.*(Si*R);                                   % Core: (x-mu)*S^-1*(x-mu)

% Fast QDA without posterior probabilities
if nargout == 1
    d = -0.5*reshape(sum(reshape(Q(:),p,m*n2),1),[m,n2])'+repmatS(-0.5*log(abs(Sa))+log(Prior'),[n2,1]);

% Full QDA with posterior values
else
    r = -0.5*reshape(sum(reshape(Q(:),p,m*n2),1),[m,n2])';

    % Alternative calculations for extreme observations, i.e. when r<-700
    q = max(r,[],2) < -700; nq = sum(q);
    if nq ~= 0
        d1 = zeros(nq,m);
        r1 = r(q,:) + repmatS(log(Prior)',[nq,1]) + repmatS(-0.5*log(abs(Sa)), [nq,1]);
        for i=1:m
            d1(:,i) = sum(exp(r1-repmatS(r1(:,i),[1,m])),2);
        end
        d(q,:) = 1./d1;
    end

    % Ordinary calculation
    PSa = repmatS(Prior'./(sqrt(abs(Sa))),[n2-nq,1]);
    d(~q,:) = PSa.*exp(r(~q,:));
    if nargout == 3
        dist = d;
    end
    d = d./repmatS(sum(d,2),[1,m]);
end

% Finds the most probable group
[~,group] = max(d,[],2);


%% Dummy function
function [dum,p] = dummy(Y)

n = size(Y,1);
p = max(Y);

dum = zeros(n,p);

for i=1:n
    dum(i,Y(i))=1;
end


%% Shorter repmat
function B = repmatS(A,siz)
[m,n] = size(A);
if (m == 1 && siz(2) == 1)
    B = A(ones(siz(1), 1), :);
elseif (n == 1 && siz(1) == 1)
    B = A(:, ones(siz(2), 1));
else
    mind = (1:m)';
    nind = (1:n)';
    mind = mind(:,ones(1,siz(1)));
    nind = nind(:,ones(1,siz(2)));
    B = A(mind,nind);
end