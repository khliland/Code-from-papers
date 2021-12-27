function sr = SR(X,beta)
%% Selectivity Ratio
% sr = SR(X,beta);

Ttp = X*(beta./norm(beta));
Xtp = Ttp*((X'*Ttp)/(Ttp'*Ttp))';
Xr  = X-Xtp;
sr  = sum(Xtp.*Xtp)./sum(Xr.*Xr);
sr(isnan(sr)) = 0;
