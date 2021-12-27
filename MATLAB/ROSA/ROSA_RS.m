function [W,P,T,Q,order,C,F] = ROSA_RS(X,Y,ncomp, cv)
% Robust scores ROSA by cross-validating candidate scores
% Single and multi-response
[n,nresp] = size(Y);
nblock = length(X);
m  = zeros(nblock,1);
Wb = cell(nblock,1);
for i=1:nblock
    X{i} = X{i}-mean(X{i});
    m(i) = size(X{i},2);
    Wb{i} = zeros(m(i),ncomp);
end
Xconcat = cell2mat(X(:)');
Y = Y-mean(Y);
W = zeros(sum(m),ncomp); T = zeros(n,ncomp);
P = zeros(size(W));      Q = zeros(nresp,ncomp);
F = zeros(nblock,ncomp); C = zeros(nblock,ncomp);
order = zeros(1,ncomp);

for a=1:ncomp
    fit = zeros(nblock,1);
    Tb  = zeros(n,nblock);
    for i=1:nblock % Application of 
        if nresp == 1
            [fit(i,1),Tb(:,i)] = cvT(X{i},Y,cv,n);
        else
            [fit(i,1),Tb(:,i)] = cvTM(X{i},Y,cv,n);
        end
    end
    [~,mc] = min(fit); order(a) = mc;
    C(:,a) = corr(Tb,Tb(:,mc))';
    F(:,a) = fit;
    if nresp == 1
        w = X{mc}'*Y; w = w./norm(w);
    else
        [w,~,~] = svds(X{mc}'*Y,1);
    end
    t = X{mc}*w;  t = t./norm(t);
    q = Y'*t;
    p = Xconcat'*t;
    w0 = zeros(sum(m),1); w0((1+sum(m(1:mc-1))):sum(m(1:mc)),1) = w;
    W(:,a) = w0; T(:,a) = t;
    Q(:,a) = q;  P(:,a) = p;
    Y = Y-t*q';
end
end

%% Cross-validated scores, mean squared deviation( t*t'*y - y )
function [m,t] = cvT(X,y,cv,n)
t = zeros(n,1);
for i=1:length(unique(cv))
    w = X(cv~=i,:)'*y(cv~=i,:);
    w = w./norm(w);
    t(cv==i,1) = X(cv==i,:)*w;
end
t = t./norm(t);
m = mean((t*(t'*y)-y).^2);
end
function [m,t] = cvTM(X,y,cv,n)
t = zeros(n,1);
for i=1:length(unique(cv))
    [w,~,~] = svds(X(cv~=i,:)'*y(cv~=i,:),1);
    w = w./norm(w);
    t(cv==i,1) = X(cv==i,:)*w;
end
t = t./norm(t);
m = mean(mean((t*(t'*y)-y).^2));
end