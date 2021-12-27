function [smcF, smcFcrit, SSCregression, SSResidual] = smc(b, X)
% [smcF smcFcrit SSCregression SSResidual] = smc(b, X)
% Input: 
%   x: Data matrix n samples x variables
%   b: PLS regression coefficients
% Output:
% smcF : SMC F-values for the list of variables
% smcFcrit: F-critical cutoff threshold value for significant important variables (smcF>smcFcrit)
%
% In case of publication of any application of this method,
% please, cite the original work:
% T.N. Tran*, N.L. Afanador, L.M.C. Buydens, L. Blanchet, 
% Interpretation of variable importance in Partial Least Squares with Significance Multivariate Correlation (sMC), 
% Chemometrics and Intelligent Laboratory Systems, Volume 138, 15 November 2014, Pages 153–160
% DOI: http://dx.doi.org/10.1016/j.chemolab.2014.08.005

alpha_mc = 0.05;

n = size(X,1);

yhat = X*b;
Xhat = (yhat*b')/(norm(b).^2);
Xresidual = X - Xhat;


SSCregression = sum(Xhat.^2);
SSResidual    = sum(Xresidual.^2);

MSCregression = SSCregression; % 1 degrees of freedom
MSResidual    = SSResidual/(n-2);

smcF     = MSCregression./MSResidual;
smcFcrit = finv(1-alpha_mc,1,n-2);

end

