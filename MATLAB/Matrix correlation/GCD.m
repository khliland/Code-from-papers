function gcd = GCD(X1,X2)
%% Yanai's GCD index
[X1,~,~] = svd(bsxfun(@minus, X1, mean(X1)), 'econ');
[X2,~,~] = svd(bsxfun(@minus, X2, mean(X2)), 'econ');

AA=X1*X1';
BB=X2*X2';

gcd = trace(AA*BB)  /(trace(AA*AA)*trace(BB*BB)).^0.5;
