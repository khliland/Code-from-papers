function VIP = vip(W, T, Q, ncomp)
% Variable importance in projections
p = size(W,1);
Q2 = Q(:)'.^2; T2 = sum(T.^2);
QT = Q2(1:ncomp).*T2(1:ncomp);
WW = W.^2; WW = WW(:,1:ncomp);
WW = WW./sum(WW);
VIP = sqrt(p * sum(WW.*QT,2)./sum(QT));
