function [Yhat, pclass, dist, YhatCV, pclassCV, distCV, nG] = LDAcv(X, Y, priors)
% (X, Y) - treningsdata
% priors - apriorisannsynligheter
% ----------------------------------
% Oppdatert 29.10-2008
% ----------------------------------

%% LDA-parametre
n     = size(X,1);           % Antall observasjoner
K     = max(Y);              % Antal klasser/grupper
Yd    = dummy(Y);            % Dummy respons
nG    = sum(Yd)';            % Gruppestørrelser
muG   = (Yd'*Yd)\Yd'*X;      % Gjennomsnitt per gruppe
XcA   = X-muG(Y,:);          % Sentrering
CovP  = XcA'*XcA./(n-K);     % Felles kovariansmatrise

%% Klargjøring
delta2 = zeros(n,K);                 % Mahalanobis
delta2CV = zeros(n,K);               % Mahalanobis CV
nn = (1:n)';                         % [1,2, ..., n]'
Xcicov = XcA/CovP;                   % Felles for alle grupper

%% Mahalanobis avstand fra xj til sentrum av gruppe k:
for k = 1:K
    XcG = X - muG(k*ones(1,n),:);
    delta2(:,k) = sum((XcG/CovP).*XcG,2);
end

%% Mahalanobis avstand fra xj til sentrum av gruppe k (kryssvalidert):
for k = 1:K
    Ydk = Yd(:,k)==1;
    Xk = sum((X-muG(k*ones(1,n),:)).*Xcicov,2);
    nGY = nG(Y);
    nGY1 = nGY(~Ydk); nGY2 = nGY(Ydk);
    delta2CV(~Ydk,k) = 1 +Xk(~Ydk,:).^2./(((n-K)*(nGY1-1)./nGY1 ...
        - delta2(nn(~Ydk)+(Y(~Ydk)-1).*n)).*delta2(~Ydk,k));
    delta2CV(Ydk,k) = (nGY2./(nGY2-1)).^2./(1-(nGY2./((nGY2-1)*(n-K))).*delta2(Ydk,k));
end


%% Klassesannsynligheter og mest sannsynlige klasse
pclass = exp((-.5)*delta2);
pclassCV = exp((-.5)*delta2CV.*delta2*((n-K-1)/(n-K)));
if nargin == 3
    pclass = pclass.*priors(ones(1,n),:);
    pclassCV = pclassCV.*priors(ones(n,1),:);
end

dist   = pclass;
distCV = pclassCV;

% Skalerer og finner så de ulike klassesannsynlighetene:
pclass = pclass./repmatS(sum(pclass,2),1,K);
pclassCV = pclassCV./repmatS(sum(pclassCV,2),1,K);

% Finner mest sannsynlige klasse:
[~, Yhat] = max(pclass,[],2);
[~, YhatCV] = max(pclassCV,[],2);


%% Kort utgave av repmat
function B = repmatS(A,M,N)
siz = [M N];
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
