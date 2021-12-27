function [A,R,U,C,predparam]=GCCA(X)
% [A,R,U,C,predparam]=GCCA(X)
% generalized canonical correlation analysis
%
% INPUT
% X         Cell array of data saisir matrices
%
% OUTPUT
% A         Cell array of canonical coefficients
% R         average corr between all pairs of canonical scores
% U         Cell array of canonical scores
% C         Matrix of consensus scores
% predparam parameters needed for prediction

ntable=length(X);

T={};
minrank=min(size(X{1}));
for i=1:ntable
    predparam.Mean{i}=mean(X{i});
    X{i}=X{i}-repmat(mean(X{i}),size(X{i},1),1);
    [U,S,V]=svd(X{i});
    if size(S,2)>1
        thisrank=rank(S);
    else
        thisrank=1;
    end
    
    if thisrank<minrank
        minrank=thisrank;
    end
    predparam.Std{i}=std(U(:,1:thisrank));
    T{i}=U(:,1:thisrank)./repmat(std(U(:,1:thisrank)),size(X{i},1),1);
    
    predparam.Pblock{i}=S(1:thisrank,1:thisrank)*V(:,1:thisrank)';
end

[C,S,V]=svd(cell2mat(T));
C=C(:,1:minrank);
predparam.Ptotal=S(1:minrank,1:minrank)*V(:,1:minrank)';

U={};
A={};
for i=1:ntable
    A{i}=pinv(X{i}'*X{i})*X{i}'*C;
    U{i}=X{i}*A{i};
end

% lambda=zeros(1,minrank);
% for j=1:minrank
% for i=1:ntable
%     lambda(j)=lambda(j)+(corr(C(:,j),U{i}(:,j)))^2;
% end
% end
% lambda=lambda/ntable

ii=0;
R=zeros(1,minrank);
for i=1:ntable
    for j=i+1:ntable
        ii=ii+1;
        R=R+diag(corr(U{i},U{j}))';
        
    end
end
R=R/ii;
