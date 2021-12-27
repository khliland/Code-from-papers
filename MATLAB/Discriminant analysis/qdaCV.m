function [group,d] = qdaCV(X,Y,varargin)

[Yd,m] = dummy(Y);  % Dummy response and max(Y)
N = sum(Yd);        % Group sizes

if isempty(varargin)
    Prior = ones(1,m)./m;
else
    Prior = varargin{1};
end
logPrior = log(Prior');

[n,p] = size(X); % dimensjonen til treningsdataene

Si = zeros(p,p,m);
Sa = zeros(1,m);

Xs = (Yd'*Yd)\Yd'*X;  % Gjennomsnitt for gruppene
Xc = X-Xs(Y,:);       % Centering
% Covariance matrices
for i=1:m
    xi = Xc(Y==i,:);
    S = xi'*xi./(N(i)-1);
    Si(:,:,i) = inv(S);
    Sa(1,i) = det(S);
end

% Fast QDA without posterior probabilities
% Common elements and declarations
d = zeros(n,m);
N12pSa = ((N-1)./(N-2)).^p.*Sa; N21  = (N-2)./(N-1);
N12    = (N-1).^2;              N012 = N./N12;
NN     = repmatS(N(Y)',[1,p]);
Xskgk  = (Xs(Y,:).*NN-X)./(NN-1); % Substitutions for group centers
if nargout == 1
    for k=1:n % Smart CV
        gk   = Y(k,1);
        Xsk  = Xs; Xsk(gk,:)   = Xskgk(k,:);
        SiXc = Xc(k,:)*Si(:,:,gk); XcSiXc = SiXc*Xc(k,:)';
        Sak  = Sa; Sak(1,gk)   = N12pSa(gk)*(1-N012(gk)*XcSiXc);
        Sik  = Si; Sik(:,:,gk) = N21(gk)*(Si(:,:,gk)+(SiXc'*N(gk)*SiXc)./(N12(gk)-N(gk)*XcSiXc));
        for j=1:m
            XXsk   = X(k,:)-Xsk(j,:);
            d(k,j) = - 0.5*XXsk*Sik(:,:,j)*XXsk' -0.5*log(abs(Sak(1,j))) + logPrior(j);
        end
    end

% Full QDA with posterior probabilities
else
    r = zeros(1,m);
    for k=1:n % Smart CV
        gk   = Y(k,1);
        Xsk  = Xs; Xsk(gk,:)   = Xskgk(k,:);
        SiXc = Xc(k,:)*Si(:,:,gk); XcSiXc = Xc(k,:)*SiXc';
        Sak  = Sa; Sak(1,gk)   = N12pSa(gk)*(1-N012(gk)*XcSiXc);
        Sik  = Si; Sik(:,:,gk) = N21(gk)*(Si(:,:,gk)+(SiXc'*N(gk)*SiXc)./(N12(gk)-N(gk)*XcSiXc));
        for j=1:m
            XXsk   = X(k,:)-Xsk(j,:);
            r(1,j) = -0.5*XXsk*Sik(:,:,j)*XXsk';
        end

        % Alternative calculation when exp(<-700)
        if max(r) <- 700
            r = r + logPrior' - 0.5*log(abs(Sak));
            for j=1:m
                d(k,j) = sum(exp(r-repmatS(r(1,j),[1,m])),2);
            end
            d(k,:) = 1./d(k,:);

        % Ordinary calculation
        else
            for j=1:m
                d(k,j) = Prior(j)/sqrt(abs(Sak(1,j)))*exp(r(1,j));
            end
        end
    end
    d = d./repmatS(sum(d,2),[1,m]);
end
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
