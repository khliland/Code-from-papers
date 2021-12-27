%% Double cross-validation with DLKPLS
% Assumes that inner cross-validation is a subset of outer cross-validation
% to reuse segments.
%		1='Full crossvalidation (leave one out)'
%		2='Random crossvalidation (samles are randomly picked for each segment)'
%		3='Systematic crossvalidation 111..-222..-333.. etc.
%		4='Systematic crossvalidation 123..-123..-123.. etc.
%       5='Vector of segments'
function [RMSECVCV,YpredsCV, RMSECV, Ypreds, ncompi] = pkplsCVCV(X,y, A, type,segments,stratify)
% ----------------------------------------------------------
% ------------------ KH Liland 2019 ------------------------
% ----------------------------------------------------------
% ------------- Solution of the PLS1-problem ---------------
% ------------- Double cross-validation style --------------

% Initializations
if nargin < 6
    stratify = [];
end
n = size(X,1);
[segsOuter, segsInner, segments] = segmentify(n, type, segments, stratify);
nseg  = max(segsOuter);
sumX = sum(X,1);
sumy = sum(y,1);
% Sum within segment (reused in inner loop)
sumSegX = (segsInner==0)*X;
sumSegy = (segsInner==0)*y;
segLength = zeros(nseg,1);
for i=1:nseg
    segLength(i) = sum(segsOuter==i);
end
% Means without i-th sample set
Xc = bsxfun(@times,bsxfun(@minus,sumX,sumSegX),1./(n-segLength));
yc = bsxfun(@times,bsxfun(@minus,sumy,sumSegy),1./(n-segLength));
m  = sum(Xc.^2,2);      % Inner products of Xc
M  = X*Xc';             % X*mean(X) for each sample mean
Ca = X*X';              % Inner product
yO = y;
Ypreds = zeros(n,A);     % Storage of predictions
n2 = false(1,n);
T  = zeros(n,A); q = T(1,:);
Ry = T;
ncompi = zeros(nseg,1);
for i=1:nseg
    % Inner cross-validation loop
    segsi = segsInner(i,:);
    nsegi = segments-1;
    ni    = length(segsi);
    Xi    = X;
    yi    = yO; yi(segsOuter==i,:) = 0;
    % Means without i-th sample set
    Xci = bsxfun(@times,bsxfun(@minus,sumX-sumSegX(i,:),sumSegX),1./(n-segLength)); 
    yci = bsxfun(@times,bsxfun(@minus,sumy-sumSegy(i,:),sumSegy),1./(n-segLength)); 
    yci(i) = [];

    m_i  = sum(Xci.^2,2);   % Inner products of Xc
    m_i(i) = [];
    M_i  = Xi*Xci';         % X*mean(X) for each sample mean
    M_i(segsOuter==i,:) = 0; M_i(:,i) = [];
    Ca_i = Ca;
    yOi  = yi;
    Ypredsi = zeros(ni,A);   % Storage of predictions
    n2i = false(1,ni); n2i(1,segsOuter==i) = true; % Lookup without outer test
    T_i  = zeros(ni,A); qi = T_i(1,:);
    Ry_i  = T_i;
    for k=1:nsegi
        inds2i = n2i;
        inds2i(segsi==k) = true;
        % Compute X*X' with centred X matrices excluding observation i
        C_i = Ca_i - (M_i(:,k) - (M_i(:,k)' + m_i(k)));
        C_i(inds2i,:) = 0;
        nti = sum(segsi==k);
        Cvt_i = Ca_i(segsi==k,segsi~=k & segsOuter~=i) - (M_i(segsi~=k & segsOuter~=i,k)' + (M_i(k,k) - m_i(k)));
        Cv_i = zeros(nti,ni);
        Cv_i(:,~inds2i) = Cvt_i;
        C_i(:,inds2i) = 0;
        yi  = yOi(:,1)-yci(k);
        yi(inds2i,:) = 0;
        for a = 1:A
            t = C_i*yi;
            if a > 1
                t = t - T_i(:,1:a-1)*(T_i(:,1:a-1)'*t);
            end
            t = t/norm(t); T_i(:,a) = t;
            % ------------------- Deflate y ----------------------
            Ry_i(:,a) = yi;
            qi(a) = yi'*t; yi = yi - qi(a)*t;
        end
        % ---------- Calculate predictions -------------
        Ypredsi(segsi==k,:) = cumsum(bsxfun(@times,((Cv_i*Ry_i)/triu(T_i'*C_i*Ry_i)), qi),2);
    end
    yci = [0;yci]; %#ok<AGROW>
    Ypredsi = bsxfun(@plus, [zeros(ni,1) Ypredsi], yci(segsi+1,:));
    RMSECVi = sqrt(((segsOuter~=i)*(bsxfun(@minus,yOi,Ypredsi).^2))./(n-segsOuter(i)));
    
    [~,ncompi(i,1)] = min(RMSECVi);
    
    % Prediction using optimal number of components
    inds2 = n2;
    inds2(segsOuter==i) = true;
    % Compute X*X' with centred X matrices excluding observation i
    C = Ca - (M(:,i) - (M(:,i)' + m(i)));
    C(inds2,:) = 0;
    nt = sum(segsOuter==i);
    Cvt = Ca(segsOuter==i,segsOuter~=i) - (M(segsOuter~=i,i)' + (M(i,i) - m(i)));
    Cv = zeros(nt,n);
    Cv(:,~inds2) = Cvt;
    C(:,inds2) = 0;
    y  = yO(:,1)-yc(i);
    y(inds2,:) = 0;
    for a = 1:A
        t = C*y;
        if a > 1
            t = t - T(:,1:a-1)*(T(:,1:a-1)'*t);
        end
        t = t/norm(t); T(:,a) = t;
        % ------------------- Deflate y ----------------------
        Ry(:,a) = y;
        q(a) = y'*t; y = y - q(a)*t;
    end
    % ---------- Calculate predictions -------------
    Ypreds(segsOuter==i,:) = cumsum(bsxfun(@times,((Cv*Ry)/triu(T'*C*Ry)), q),2);
end    

Ypreds = bsxfun(@plus, [zeros(n,1) Ypreds], yc(segsOuter,:));
RMSECV = sqrt(mean(bsxfun(@minus,yO,Ypreds).^2));

YpredsCV = zeros(n,1);
for i=1:nseg
    YpredsCV(segsOuter==i,1) = Ypreds(segsOuter==i, ncompi(i));
end
RMSECVCV = sqrt(mean(bsxfun(@minus,YpredsCV,yO).^2));


%% Create segment combinations                                             ------ TODO: should inner segments be a subset of outer segments? -------
function [segsOuter, segsInner, segments] = segmentify(N, type, segments, stratify)
% N    : number of samples
% type - type of crossvalidation
%		1='Full crossvalidation (leave one out)'
%		2='Random crossvalidation (samles are randomly picked for each segment)'
%		3='Systematic crossvalidation 111..-222..-333.. etc.
%		4='Systematic crossvalidation 123..-123..-123.. etc.
%       5='Vector of segments'
% segments - number of segments
% stratify - vector of classes to stratify over
% 
% segsOuter - vector of indices
% segsInner - cell of vectors of indices assuming removal of chosen outer
if type == 1 % LOO
    segsOuter = 1:N;
    segments  = N;
elseif type == 2 % Random segments
    if ~isempty(stratify)
        stop('Not implemented yet')
    else
        segsOuter = repmat(1:segments,1,ceil(N/segments));
        r  = randperm(N);
        segsOuter = segsOuter(r);
    end
elseif type == 3 % Consecutive
    if ~isempty(stratify)
        stop('Not implemented yet')
    else
        segsOuter = repmat(1:segments,1,ceil(N/segments)); segsOuter = sort(segsOuter(1:N));
    end
elseif type == 4 % Interleaved
    if ~isempty(stratify)
        stop('Not implemented yet')
    else
        segsOuter = repmat(1:segments,1,ceil(N/segments)); segsOuter = segsOuter(1:N);
        segsOuter = segsOuter(1:N);
    end
elseif type == 5 % User chosen
    if ~isempty(stratify)
        stop('Not implemented yet')
    else
        if size(segments,2) == 1
            segsOuter = segments';
        else
            segsOuter = segments;
        end
        segments  = max(segsOuter);
    end
end
segsInner = zeros(segments,N);
for i=1:segments
    segs = segsOuter;
    segs(segs==i) = 0;
    segs(segs>i)  = segs(segs>i)-1;
    segsInner(i,:) = segs;
end
