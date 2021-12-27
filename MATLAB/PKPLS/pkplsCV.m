function [RMSECV, Ypreds] = pkplsCV(X,y,A,cv)
% -----------------------------------------------------
% ---------------- KH Liland 2019 ---------------------
% -----------------------------------------------------
%- --------- Solution of the PLS1 CV-problem ----------
nseg  = max(cv);
if size(cv,2) == 1
    cv = cv';
end
n = size(X,1);
sumX = sum(X,1);
sumy = sum(y,1);
segLength = zeros(nseg,1);
cvMat = zeros(nseg,n);
for i=1:nseg
    segLength(i) = sum(cv==i);
    cvMat(i,cv==i) = 1;
end
% Means without i-th sample set
Xc = bsxfun(@times,bsxfun(@minus,sumX,cvMat*X), ...
    1./(n-segLength)); 
yc = bsxfun(@times,bsxfun(@minus,sumy,cvMat*y), ...
    1./(n-segLength));
m  = sum(Xc.^2,2);     % Inner products of Xc
M  = X*Xc';            % X*mean(X) for each sample mean
Ca = X*X';             % Inner product
yO = y;                % Original response
Ypreds = zeros(n,A);   % Storage of predictions
n2 = false(1,n);
T  = zeros(n,A); q = T(1,:);
Ry = T;
for i=1:nseg
    inds2 = n2;
    inds2(cv==i) = true;
    % Compute X*X' with centred X matrices 
    % excluding observation set i
    C = Ca - (M(:,i) - (M(:,i)' + m(i)));
    C(inds2,:) = 0;
    nt = sum(cv==i);
    % Compute Xval*Xval' with centred X matrices 
    % excluding observation set i
    Cvt = Ca(cv==i,cv~=i) - (M(cv~=i,i)' + ...
        (M(i,i) - m(i)));
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
        % ---------------- Deflate y ------------------
        Ry(:,a) = y;
        q(a) = y'*t; y = y - q(a)*t;
    end
    % ---------- Calculate predictions -------------
    Ypreds(cv==i,:) = cumsum(bsxfun(@times, ...
        ((Cv*Ry)/triu(T'*C*Ry)), q),2);
end
% Predicted 0-th component = mean(y) (per segment)
Ypreds = bsxfun(@plus, [zeros(n,1) Ypreds], yc(cv,:));
RMSECV = sqrt(mean(bsxfun(@minus,yO,Ypreds).^2));