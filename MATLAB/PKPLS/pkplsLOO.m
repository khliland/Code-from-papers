function [RMSECV, Ypreds] = pkplsLOO(X,y,A)
% -----------------------------------------------------
% ----------------- KH Liland 2019 --------------------
% -----------------------------------------------------
% --------- Solution of the PLS1 LOO-problem ----------
n  = size(X,1);
Xc = (sum(X)-X)./(n-1);% Means without i-th sample
m  = sum(Xc.^2,2);     % Inner products of Xc
M  = X*Xc';            % X*mean(X) for each sample mean
Ca = X*X';             % Inner product
yO = y;                % Original response
T  = zeros(n,A); q = T(1,:);
Ry = T;
yc = (sum(y,1)-y)./(n-1); % Means without i-th sample
Ypreds = zeros(n,A);      % Storage of predictions
n2 = false(1,n);
for i=1:n
    % Compute X*X' with centred X matrices 
    % excluding observation i
    inds2 = n2;
    inds2(i) = true;
    C = Ca - (M(:,i) + (M(:,i)' - m(i)));
    C(inds2,:) = 0;
    % Compute Xval*Xval' with centred X matrices 
    % excluding observation i
    Cv = C(:,i)';
    C(:,inds2) = 0;
    y = yO-yc(i);
    for a=1:A
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
    Ypreds(i,:) = cumsum(bsxfun(@times, ...
        ((Cv*Ry)/triu(T'*C*Ry)), q),2);
end
% Predicted 0-th component = mean(y) (per segment)
Ypreds = bsxfun(@plus, [zeros(n,1) Ypreds], yc);
RMSECV = sqrt(mean(bsxfun(@minus,yO,Ypreds).^2));