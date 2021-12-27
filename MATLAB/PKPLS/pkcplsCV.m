function [RMSECV, Ypreds] = pkcplsCV(X,y,A,cv,yadd,cca)
% ----------------------------------------------------------
% ------------------ KH Liland 2019 ------------------------
% ----------------------------------------------------------
% ----------- Solution of the (C)PLS(2) CV-problem ---------

% Initializations
nseg  = max(cv);
if size(cv,2) == 1
    cv = cv';
end
n = size(X,1); nresp = size(y,2);
segLength = zeros(nseg,1);
cvMat = zeros(nseg,n);
for i=1:nseg
    segLength(i) = sum(cv==i);
    cvMat(i,cv==i) = 1;
end
if nargin < 5 || isempty(yadd)
    yaddO = []; yadd = [];
else
    yaddO     = yadd;
    ycadd = zeros(nseg,nresp);
    for i=1:nseg
        ycadd(i,:)  = (sum(yadd,1)-sum(yadd(cv==i,:),1))/(n-segLength(i));
    end
end
sumX = sum(X,1);
sumy = sum(y,1);
Xc = bsxfun(@times,bsxfun(@minus,sumX,cvMat*X),1./(n-segLength)); % Means without i-th sample set
yc = bsxfun(@times,bsxfun(@minus,sumy,cvMat*y),1./(n-segLength)); % Means without i-th sample set
m  = sum(Xc.^2,2);     % Inner products of Xc
M  = X*Xc';            % X*mean(X) for each sample mean
Ca = X*X';             % Inner product
yO = y;                % Original responses
T  = zeros(n,A);  Q = zeros(A, nresp);
Ry = T;
Ypreds = zeros(n,A,nresp);    % Storage of predictions
if nargin < 6
    cca = false;
end
n2 = false(1,n);
for i=1:nseg
    inds2 = n2;
    inds2(cv==i) = true;
    % Compute X*X' with centred X matrices excluding observation i
    C = Ca - (M(:,i) + (M(:,i)' - m(i)));
    C(inds2,:) = 0;
    nt = sum(cv==i);
    % Compute Xval*Xval' with centred X matrices excluding observation i
    Cvt = Ca(cv==i,cv~=i) - (M(cv~=i,i)' + (M(i,i) - m(i)));
    Cv = zeros(nt,n);
    Cv(:,~inds2) = Cvt;
    C(:,inds2) = 0;
    y = bsxfun(@minus,yO,yc(i,:));
    y(inds2,:) = 0;
    Co = C;
    if ~isempty(yaddO)
        yadd = bsxfun(@minus,yaddO,ycadd(i,:));
    end
    for a = 1:A
        if cca
            tt = C*[y yadd];
            [~,w] = ccXY(tt,y,[]); % CPLS
        else
            tt = C*y;
            [~,~,w] = svd(tt,'econ'); % PLS2
        end
        t = tt*w(:,1);
        if a > 1
            t = t - T(:,1:a-1)*(T(:,1:a-1)'*t);
        end
        t = t/norm(t); T(:,a) = t;
        % ------------------- Deflate y ----------------------
        Ry(:,a) = [y yadd]*w(:,1);
        Q(a,:) = t'*y; y = y - t*Q(a,:);
        % ------------------- Deflate C ---------------------
        if cca % ... when using CPLS
            Ctt = (C*t)*t';
            C = C - Ctt - Ctt' + t*(t'*Ctt);
        end
    end
    % ---------- Calculate predictions -------------
    Ypreds(cv==i,:,:) = cumsum(bsxfun(@times,((Cv*Ry)/triu(T'*Co*Ry)), reshape(Q, [1,A,nresp])),2);
end
Ypreds = bsxfun(@plus, cat(2, zeros(n,1,nresp), Ypreds), reshape(yc(cv,:), [n,1,nresp]));
RMSECV = squeeze(sqrt(mean(bsxfun(@minus,reshape(yO,[n,1,nresp]),Ypreds).^2, 1)));


%% Canonical correlations
function [r,A] = ccXY(X,Y,wt)
% Computes the coefficients in canonical variates between collumns of X and Y
[n,p1] = size(X); p2 = size(Y,2);

% Weighting of observations with regards to wt (asumes weighted centering already performed)
if ~isempty(wt)
    X = rscale(X,wt);
    Y = rscale(Y,wt);
end

% Factoring of data by QR decomposition and ellimination of internal linear
% dependencies in X and Y
[Q1,T11,perm1] = qr(X,0);       [Q2,T22,~] = qr(Y,0);
rankX          = sum(abs(diag(T11)) > eps(abs(T11(1)))*max(n,p1));
rankY          = sum(abs(diag(T22)) > eps(abs(T22(1)))*max(n,p2));
if rankX < p1
    Q1 = Q1(:,1:rankX); T11 = T11(1:rankX,1:rankX);
end
if rankY < p2
    Q2 = Q2(:,1:rankY);
end

% Economical computation of canonical coefficients and canonical correlations
d = min(rankX,rankY);
if nargout == 1
    D    = svd(Q1' * Q2,0);
    r    = min(max(D(1:d), 0), 1); % Canonical correlations
else
    [L,D]    = svd(Q1' * Q2,0);
    A        = T11 \ L(:,1:d) * sqrt(n-1);
    % Transform back coefficients to full size and correct order
    A(perm1,:) = [A; zeros(p1-rankX,d)];
    r = min(max(diag(D(1:d)), 0), 1); % Canonical correlations
end
