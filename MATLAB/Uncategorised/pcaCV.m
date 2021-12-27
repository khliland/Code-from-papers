function [error] = pcaCV(X, ncomp, progress)

X = bsxfun(@minus,X,mean(X));
[n,p] = size(X);
if n<p % Transpose SVD solution for efficiency
    X = X';
    transp = true;
else
    transp = false;
end
ncomp = min(ncomp,p);
Xhat  = zeros(n,p,ncomp);

if nargin < 3 || progress == 1
    progress = 1;
    inds = round((1:n/9:n));
    pr   = 10:10:100; k = 1;
    h = waitbar(0,'0%','Name','Estimating error ...');
end
trues = true(1,n);
for i=1:n
    if progress == 1 && any(inds==i)
        waitbar((i-1)/n, h, [num2str(pr(k)) '%']);
        k = k+1;
    end
    tr = trues; tr(1,i) = false;
    if transp
        Xi  = X(:,tr); % Leave out sample
        [Pi,~,~] = svd(Xi,'econ');
        Xii = repmat(X(:,i)',p,1);
    else
        Xi  = X(tr,:); % Leave out sample
        [~,~,Pi] = svd(Xi,'econ'); 
        Xii = repmat(X(i,:),p,1);
    end
    for j=1:p
        Xii(j,j) = 0;
    end
    Pi  = Pi(:,1:ncomp);       % P from X without sample i

    PiP  = cumsum(Pi'.^2)';
    PiP1 = PiP./(1-PiP)+1;
    PihP = Pi'.*(Xii*Pi)';
    for j=1:p
        Xhat(i,j,:) = sum(triu(PihP(:,j)*PiP1(j,:)));
    end
end
if progress == 1
    waitbar(1, h, '100%');
end
if transp
    X = X';
end
error = zeros(ncomp,1);
for i=1:ncomp
    error(i,1) = sum(sum((X-Xhat(:,:,i)).^2));
end
if progress == 1
    delete(h)
end
