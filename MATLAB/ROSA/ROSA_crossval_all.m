function [RMSECV,orders] = ROSA_crossval_all(X,Y,A, nseg, cvtype)

n  = size(X{1},1);
nb = length(X);
if cvtype~=5
    [cv, nseg] = cvseg(n,nseg,cvtype);
else
    cv = nseg;
    nseg = max(cv);
end

for i=1:nseg
    Xin  = cell(1,nb);
    Xout = cell(1,nb);
    for a=1:nb
        Xin{a}  = X{a}(cv~=i,:);
        Xout{a} = X{a}(cv==i,:);
    end
    Yin  = Y(cv~=i,:);
    Yout = Y(cv==i,:);
    
    [SSEPi, orders] = ROSA_rec(Xin,Yin, Xout,Yout, A);
    if i==1
        SSEP = SSEPi;
    else
        SSEP = SSEP + SSEPi;
    end
end
RMSECV = sqrt(SSEP./n);

