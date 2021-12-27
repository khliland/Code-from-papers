function [RV, RV2, RVadj] = RV_modified(X,Y)
AA=X*X';
BB=Y*Y';
AA0 = AA - diag(diag(AA),0);
BB0 = BB - diag(diag(BB),0);

RV  = trace(AA*BB)  /(trace(AA*AA)*trace(BB*BB)).^0.5;
RV2 = trace(AA0*BB0)/(ssq(AA0)^.5*ssq(BB0)^.5);

n = size(X,1);

p = size(X,2); q = size(Y,2); pq = p*q; pp = p*p; qq = q*q;
sx = std(X); sy = std(Y); msxy = [min(sx) max(sx) min(sy) max(sy)];
if any(msxy > 1+10^-12) || any(msxy < 1-10^-12) % Not standardized X/Y
    Xs = bsxfun(@times,X,1./sx); % Standardize
    Ys = bsxfun(@times,Y,1./sy); % --- || ---
    AAs = Xs*Xs';
    BBs = Ys*Ys';
    % Find scaling between R2 and R2adj
    xy = trace(AAs*BBs)/(pq-(n-1)/(n-2)*(pq-trace(AAs*BBs)/(n-1).^2));
    xx = trace(AAs*AAs)/(pp-(n-1)/(n-2)*(pp-trace(AAs*AAs)/(n-1).^2));
    yy = trace(BBs*BBs)/(qq-(n-1)/(n-2)*(qq-trace(BBs*BBs)/(n-1).^2));
    % Apply scaling to non-standarized data
    RVadj = (trace(AA*BB)/xy)  /(trace(AA*AA)/xx*trace(BB*BB)/yy).^0.5;
else
    RVadj = (pq-(n-1)/(n-2)*(pq-trace(AA*BB)/(n-1).^2))/sqrt((pp-(n-1)/(n-2)*(pp-trace(AA*AA)/(n-1).^2))*(qq-(n-1)/(n-2)*(qq-trace(BB*BB)/(n-1).^2)));
end

function ss = ssq(X)
ss = sum(X(:).^2);

