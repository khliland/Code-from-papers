function [error, Ypreds] = pkplsmCV(X,Y, A,cv, progress)
% ----------------------------------------------------------
% ------------------ KH Liland 2019 ------------------------
% ----------------------------------------------------------
% ------------- Solution of the PLS1 CV-problem ------------
% ---------- for multiple responses, e.g. dummy Y ----------

% If Y is single column, it will be converted to a dummy matrix, and
% the subsequent analysis becomes classification.

if nargin < 5
    progress = true;
end
Yo = Y; % Original Y
if size(Y,2) == 1
    Y = dummy(Y);
    classify = true;
else
    classify = false;
end
nresp = size(Y,2);

% Initializations
nseg  = max(cv); cv = cast(cv,'int16');
if size(cv,2) == 1
    cv = cv';
end
n = size(X,1);
sumX = sum(X,1);
sumY = sum(Y,1);
segLength = zeros(nseg,1, 'like',X);
cvMat = zeros(nseg,n, 'like',X);
for i=1:nseg
    segLength(i) = sum(cv==i);
    cvMat(i,cv==i) = 1;
end
Xc = bsxfun(@times,bsxfun(@minus,sumX,cvMat*X),1./(n-segLength)); % Means without i-th sample set
Yc = bsxfun(@times,bsxfun(@minus,sumY,cvMat*Y),1./(n-segLength)); % Means without i-th sample set
m  = sum(Xc.^2,2);     % Inner products of Xc
M  = X*Xc';            % X*mean(X) for each sample mean
if progress
    f = waitbar(0,'1','Name','Multi-response PLS...',...
        'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
    setappdata(f,'canceling',0);
    waitbar(0,f,'Calculating inner product')
end
Ca = X*X';             % Inner product
Ypreds = zeros(n,nresp,A, 'like',X);         % Storage of predictions
n2  = false(1,n);
for i=1:nseg
    if progress
        if getappdata(f,'canceling')
            break
        end
        waitbar(i/(nseg+1),f,['Segment ' num2str(i) '/' num2str(nseg)])
    end
    inds2 = n2;
    inds2(cv==i) = true;
    % Compute X*X' with centred X matrices excluding observation set i
    C = Ca - (M(:,i) - (M(:,i)' + m(i)));
    C(inds2,:) = 0;
    nt = sum(cv==i);
    % Compute Xval*Xval' with centred X matrices excluding observation set i
    Cvt = Ca(cv==i,cv~=i) - (M(cv~=i,i)' + (M(i,i) - m(i)));
    Cv = zeros(nt,n, 'like',X);
    Cv(:,~inds2) = Cvt;
    C(:,inds2) = 0;
    % Don't loop over responses
    y  = Y-Yc(i,:);
    y(inds2,:) = 0;
    T = zeros(n,A,nresp, 'like',X); Ry = T;
    q = zeros(nresp,A, 'like',X);
    for a = 1:A
        if progress
            if getappdata(f,'canceling')
                break
            end
            waitbar(i/(nseg+1)+(1/(nseg+1))/A*(a-1),f,['Segment ' num2str(i) '/' num2str(nseg) ', component ' num2str(a) '/' num2str(A)])
        end
        t = C*y;
        if a > 1
            for resp=1:nresp
                sT = T(:,1:a-1,resp);
                tr = t(:,resp);
                t(:,resp) = tr - sT*(sT'*tr);
            end
        end
        t = bsxfun(@times,t,1./norms(t)); 
        T(:,a,:) = t;
        % ------------------- Deflate y ----------------------
        Ry(:,a,:) = y;
        q(:,a) = sum(y.*t); y = y - q(:,a)'.*t;
    end
    % ---------- Calculate predictions -------------
    RyUF = reshape(Ry,[n,A*nresp]);
    XXvRy = Cv*RyUF; XXRy = C*RyUF;
    for resp=1:nresp
        sT = T(:,:,resp);
        Ypreds(cv==i,resp,:) = cumsum(bsxfun(@times,XXvRy(:,(resp-1)*A+(1:A))/triu(sT'*XXRy(:,(resp-1)*A+(1:A))), q(resp,:)),2);
    end
end
if progress
    waitbar(1,f,'Finishing up...')
end    
if classify
    Ypreds = bsxfun(@plus, cat(3,zeros(n,nresp, 'like',X), Ypreds), Yc(cv,:));
    error = zeros(1,A, 'like',X);
    for i=1:A
        [~,max_i] = max(Ypreds(:,:,i),[],2);
        error(1,i) = sum(max_i~=Yo)/n;
    end
else
    Ypreds = bsxfun(@plus, cat(3,zeros(n,nresp, 'like',X), Ypreds), Yc(cv,:));
    error = squeeze(sqrt(mean(bsxfun(@minus,Yo,Ypreds).^2)));
end
if progress
    delete(f)
end