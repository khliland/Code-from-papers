function [RMSECV, Ypreds] = pkpcrLOO(A,X,y)
% -----------------------------------------------------
% ---------------- KH Liland 2019 ---------------------
% -----------------------------------------------------
% -------- Solution of the PCA/R LOO-problems ---------
n  = size(X,1);
Xc = (sum(X)-X)./(n-1); % Means without i-th sample
m  = sum(Xc.^2,2);     % Inner products of Xc
M  = X*Xc';            % X*mean(X) for each sample mean
Ca = X*X';             % Inner product
if nargin == 3
    yO  = y;
    yc  = (sum(y,1)-y)./(n-1); % Means without i-th obs.
end
Ypreds = zeros(n,A);   % Storage of predictions
n2 = false(1,n);
for i=1:n
    % Compute X*X' with centred X matrices 
    % excluding observation i
    inds2 = n2;
    inds2(i) = true;
    C = Ca - (M(:,i) + (M(:,i)' - m(i)));
    if nargin == 2
        Cvv = C(inds2,inds2);
    end
    C(inds2,:) = 0;
    % Compute Xval*Xval' with centred X matrices 
    % excluding observation i
    Cv = C(:,i)';
    C(:,inds2) = 0;
    
    [U,s] = eig(C, 'vector');
    s = s(end:-1:(end-A+1))';
    U = U(:,end:-1:(end-A+1));
    if nargin == 3
        % Perform PCR
        y = yO-yc(i);
        Ypreds(i,:) = cumsum((Cv*U).*((y'*U)./s),2);
    else
        % Perform PCA
        Ypreds(i,:) = trace(Cvv)- ...
            cumsum(sum((Cv*U).^2,1)./s); 
    end
end
if nargin == 3
    Ypreds = bsxfun(@plus, [zeros(n,1) Ypreds], yc);
    RMSECV = sqrt(mean(bsxfun(@minus,yO,Ypreds).^2));
else
    RMSECV = sqrt(mean(Ypreds));
end