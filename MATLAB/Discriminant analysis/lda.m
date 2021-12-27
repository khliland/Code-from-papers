function [group,d,dist] = lda(X,Y,varargin)
% Linear discriminant analysis
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

[Yd,m] = dummy(Y); % Dummy respons and number of classes
m = m+1;
Y = Yd*(1:m)';

if isempty(varargin)
    Xnew  = X;
    Prior = ones(m,1)./m;
elseif length(varargin) == 1
    Xnew  = varargin{1};
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
[n,p] = size(X);
n2    = size(Xnew,1);

Xs = (Yd'*Yd)\Yd'*X;   % Mean for each group
Xc = X-Xs(Y,:);        % Centering
S  = Xc'*Xc./(n-m);    % Common covariance matrix

% Fast LDA without posterior probabilities
if nargout == 1
    % Common element across all observations
    R1 = Xs/S;  % '/' is faster, 'pinv' more robust
    R2 = - 0.5*diag(R1*Xs') + log(Prior);
    d  = Xnew*R1' + repmatS(R2',[n2,1]);

% Full LDA with posterior values
else
    Sib = zeros(m*p,m*p); % Block diagonal sparse matrix contianing all Si
    Si  = inv(S);         % 'inv' is faster, 'pinv' more robust
    for i=1:m
        Sib((i-1)*p+(1:p),(i-1)*p+(1:p)) = Si;
    end
    Xst = Xs';                                       % Transposed means
    R = repmatS(Xnew',[m,1])-repmatS(Xst(:),[1,n2]); % Center using all groups
    Q = R.*(Sib*R);                                  % Core: (x-mu)*S^-1*(x-mu)

    d = zeros(n2,m);
    r = -0.5*reshape(sum(reshape(Q(:),p,m*n2),1),[m,n2])';
    
    % Alternative calculations for extreme observations, i.e. when r<-700
    q = max(r,[],2) < -700; 
    if any(q)
        nq = sum(q);
        d1 = zeros(nq,m);
        r1 = r(q,:) + repmatS(log(Prior'),[nq,1]);
        for i=1:m
            d1(:,i) = sum(exp(r1-repmatS(r1(:,i),[1,m])),2);
        end
        d(q,:) = 1./d1;
    end

    % Ordinary calculation
    PSa = repmatS(Prior',[n2-sum(q),1]);
    d(~q,:) = PSa.*exp(r(~q,:));
    if nargout == 3
        dist = d;
    end
    d = d./repmatS(sum(d,2),[1,m]);
end

[~,group] = max(d,[],2);  % Finds the most probable group


%% Dummy function
% function [dum,p] = dummy(Y)
% 
% n = size(Y,1);
% p = max(Y);
% 
% dum = zeros(n,p);
% 
% for i=1:n
%     dum(i,Y(i))=1;
% end
function [dY,m] = dummy(Y)
% Rutinen konverterer en vektor Y
% av K (uordnede) klasselabler 
% til en N*K indikatormatrise
Ys  = sort(Y(:));
dYs = [0;diff(Ys)];
s   = Ys(dYs~=0); % Tilsvarer sort(unique(Y))
m   = length(s);

dY = zeros(size(Y,1),m+1);
dY(:,1) = Y==Ys(1);
for i = 1:m
    dY(:,i+1) = Y==s(i);
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