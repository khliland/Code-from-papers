%      results = ffmanova(X,Y,cova,model,xNames,stand,nSim,Xnew)
%  or  results = ffmanova(    modelFormula     ,stand,nSim,Xnew)
%    Performs general linear modelling of several response variables (Y). 
%    Collinear and highly correlated response variables are handled. 
%    The X-factors can be categorical, continuous and composite continuous.
%
%     The function calculates
%     - 50-50 MANOVA results.
%     - raw single response p-values.
%     - familywise adjusted and false discovery rate adjusted single 
%       response p-values by rotation testing.
%     - predictions, mean predictions and least squares means.
%     - standard deviations of those predictions. 
%
% ==========   INPUT ==========:
%      X{1,#Xvariables} - design information as cell array. Categorical design variables 
%            can be represented by a numeric vector, a numeric matrix (each unique row 
%            is a group), a character matrix (each row representing a group name), or 
%            a cell array of strings stored as a column vector. Nonzero elements of 
%            cova indicate cells of X that are covariates. Multiple column of covariate 
%            model terms are allowed. 
%            - Alternatively X can be an ordinary matrix where each column
%            is a design variable.
%      Y(#observations,#responses) - matrix of response values. 
%              cova(1,#Xvariables) - covariate terms (see above)
%        model(#terms,#Xvariables) - model matrix or order coded model or 
%                                    text coded model (see below)
%                           stand  - standardization of responses, = 0 (default) or 1
%                           xNames - Names of x factors. Default: {'A' 'B' 'C'}
%                   nSim(1,#terms) - Number of rotation testing simulations.
%                            Xnew  - cell array of cell arrays Xnew = {Xnew1; Xnew2; ..},
%                                    new X's for prediction calculations.
%                            cXnew - cell array cXnew = {cXnew1(*,*); cXnew2(*,*); ..},
%                                    Predicts linear combinations (default: identity matrix) 
%                                       cXnew*Xnew
%                          nSimXNew - When cXnew and nSimXNew are specified:
%                                       Significance tests according to cXnew*Xnew
%                                       50-50 MANOVA results 
%                                       + rotation tests(when nSimXNew>0)
%
%   NOTE:                                 
%       - Some cells of Xnew1 (and Xnew2...) can be empty ("[]") - leading
%              to mean predictions and least squares means.
%       - nSim can be a single number -> equal nSim for all terms.
%       - nSim =  0 -> pAdjusted and pAdjFDR are not calculated.
%       - nSim = -1 -> pRaw, pAdjusted and pAdjFDR are not calculated.
%       - This is similar for nSimXNew
%       - default cova is [0 0 0 ...]
%       - default Y is zeros(#observations,1)
%
%   MODEL CODING:
%       - order coded model:
%             model{1,#Xvariables} specifys maximum order of the factors
%       - text coded model:
%              'linear'    is equivalent to { 1 1 1 ..... 1}
%              'quadratic' is equivalent to { 2 2 2 ..... 2}
%              'cubic'     is equivalent to { 3 3 3 ..... 3}                    
%       - model matrix example: X = {A B C}
%                model = [1 0 0; 0 1 0 ; 0 0 1; 2 0 0; 1 1 0; 1 0 1; 0 1 1; 3 0 0]
%                 ->   Constant + A + B + C + A^2 + A*B + A*C + B*C + A^3
%           Constant term is automatically included. But create constant term 
%           manually ([0 0 0; ...]) to obtain constant term output.
%       - default model is the identity matrix -> main factor model
%                 
%         When X or Y is empty the model matrix is returned and printet (with matlab code)
%                examples: model = manova5050([],[],[0 1 0 1],{3 2 1 3});
%                          model = manova5050([],[],[0 1 0 1],'quadratic');
%
%  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
%  :::::    USING "ModelFormula" instead of "X,Y,cova,model,xNames"   :::::::
%  ::
%  ::    ffmanova('Y = A + B + A*B + C|D|E + F|G|H@2 + I|J|K#2 + L^3 + M#3 + N#4 - N^3')
%  ::        givs this model:   A    B    A*B    C    D    E    C*D  C*E  D*E  C*D*E
%  ::          F    G    H    F*G  F*H  G*H  
%  ::          I    J    K    I*J  I*K  J*K  I^2  J^2  K^2  
%  ::          L^3    M    M^2  M^3   N    N^2  N^4
%  ::
%  ::      @2 means interactions up to order 2
%  ::      #2 means all terms up to order 2 
%  :: 
%  ::      A variable is treated as categorical if $ is included at the end 
%  ::      of the variable name (anywhere in a complex model formula).
%  ::      A variable that is cell array is treated as categorical (A->{A}).
%  ::
%  ::      Except that =,+,-,|,@,#,*,^ are special symbols in the model formula,
%  ::      ffmanova uses eval to interpret the string. 
%  ::      ffmanova('log(100+Y) = a + b==2 + 1./c')  is a valid expression.
%  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
%
%
% ==========   OUTPUT   ========== results is a structure with fields:
%     termNames: name of model terms (including "error").
%       exVarSS: (Sum of SS for each response)/(Sum of total SS for each response).
%            df: degrees of freedom - adjusted for other terms in model.
%         df_om: degrees of freedom - adjusted for terms contained in actual term.
%           nPC: number of principal components used for testing.
%           nBU: number of principal components used as buffer components.
%       exVarPC: variance explained by nPC components
%       exVarBU: variance explained by (nPC+nBU) components
%       pValues: 50-50 MANOVA p-values.
%    outputText: 50-50 MANOVA results as text.
%          Yhat: Fitted values. 
%       YhatStd: Standard deviations of the fitted values.
%          nSim: as input (-1 -> 0), but could have been changed interactively.
%     pAdjusted: familywise adjusted p-values.
%       pAdjFDR: false discovery rate adjusted p-values.
%          pRaw: raw p-values.
%          stat: Unvivariate t-statistics (df=1) or  F-statistics (df>1)
%       newPred: Yhat's and YhatStd's according to Xnew
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%    Copyright, Oyvind Langsrud, MATFORSK, 2005 %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [results,Y,cova,model,xNames]=ffmanova(varargin)
if ischar(varargin{1})
    switch  varargin{1}
        case 'continue'           
            siminfo(varargin{1}); 
            return;               
        case 'pause'
            siminfo(varargin{1});
            return;
        case 'stop'
            siminfo(varargin{1});
            return;
        otherwise
            i=1;
            while i<= nargin 
                switch i
                    case 1
                        s_model = varargin{i};
                    case 2
                        stand = varargin{i};
                    case 3
                        nSim = varargin{i};
                    case 4
                        Xnew = varargin{i};
                    case 5
                        cXnew = varargin{i};
                    case 6
                        nSimXNew = varargin{i};
                end     
                i=i+1;
            end
    end
else
    i=1;
    while i<= nargin 
        switch i
            case 1
                X = varargin{i};
            case 2
                Y = varargin{i};
            case 3
                cova = varargin{i};
            case 4
                model = varargin{i};
            case 5
                xNames = varargin{i};
            case 6
                stand = varargin{i};
            case 7
                nSim = varargin{i};
            case 8
                Xnew = varargin{i};
            case 9
                cXnew = varargin{i};
            case 10
                nSimXNew = varargin{i};
        end
        i=i+1;
    end
    switch nargin
        case 1
            results = manova5050nostring(X);
        case 2
            results = manova5050nostring(X,Y);
        case 3
            results = manova5050nostring(X,Y,cova);
        case 4
            results = manova5050nostring(X,Y,cova,model);
        case 5
            results = manova5050nostring(X,Y,cova,model,xNames);
        case 6
            results = manova5050nostring(X,Y,cova,model,xNames,stand);
        case 7
            results = manova5050nostring(X,Y,cova,model,xNames,stand,nSim);
        case 8
            results = manova5050nostring(X,Y,cova,model,xNames,stand,nSim,Xnew);
        case 9
            results = manova5050nostring(X,Y,cova,model,xNames,stand,nSim,Xnew,cXnew);
        case 10
            results = manova5050nostring(X,Y,cova,model,xNames,stand,nSim,Xnew,cXnew,nSimXNew);
    end
    return;
end
try
    [y,pos,pow,cpow,dpow,mult,cat,e,f]=smodel_(s_model);
catch
    error(lasterr);
end
try
    Y = evalin('caller',y);
catch
    err_s = sprintf('"%s" not interpretable.',y);
    error(err_s);
end
cova = 1-cat;
nVar = length(f);
X = cell(1,nVar);
for i=1:nVar
    try
        X{i} = evalin('caller',f{i});
    catch
        err_s = sprintf('"%s" not interpretable. Maybe "+" is missing?',f{i});
        error(err_s);
    end
    if ~isnumeric(X{i})
        cova(i)=0;
        if iscell(X{i})
            if length(X{i})==1
                X{i} = X{i}{1};
            end
        end
    end
end
xNames = f;
model=zeros(0,nVar);
for i=1:length(e)
    if (length(e{i})==1 & cpow(i)==0) | mult(i)   
        m=zeros(1,nVar);
        for j=1:length(e{i})
            m(e{i}(j)) = m(e{i}(j)) +1;
        end
        if dpow(i)
            m=m*dpow(i);
        end
    else  
        order = zeros(1,nVar);
        order(e{i}) = 1;
        if(cpow(i))  
            order = order*cpow(i);
            m = modelmatrix(cova,m2c(order),xNames,1);
        else
            if(pow(i))
                order = order*pow(i);
            else      
                order = order*sum(order);
            end
            m = modelmatrix(zeros(1,nVar),m2c(order),xNames,1);
        end
    end
    if pos(i) 
        sm1_ = size(model,1);
        model = [model' m']';
        sm1 = size(model,1);
        ind = ones(sm1,1)==1;
        for j=(sm1_+1):sm1
            m_=ones(sm1,1)*model(j,:);
            ind(j) =min(find(sum((m_~=model),2)==0))>=j;
        end
        model = model(ind,:);
    else   
        sm1 = size(model,1);
        ind = ones(sm1,1)==1;
        for j=1:size(m,1);
            m_=ones(sm1,1)*m(j,:);
            ind(sum((m_~=model),2)==0)=0;
        end
        model = model(logical(ind),:);
    end
end
if nargout==5
    results=X;
    return;
end
switch nargin
    case 1
        results = manova5050nostring(X,Y,cova,model,xNames);
    case 2
        results = manova5050nostring(X,Y,cova,model,xNames,stand);
    case 3
        results = manova5050nostring(X,Y,cova,model,xNames,stand,nSim);
    case 4
        results = manova5050nostring(X,Y,cova,model,xNames,stand,nSim,Xnew);
    case 5
        results = manova5050nostring(X,Y,cova,model,xNames,stand,nSim,Xnew,cXnew);
    case 6
        results = manova5050nostring(X,Y,cova,model,xNames,stand,nSim,Xnew,cXnew,nSimXNew);
end
function [y,pos,pow,cpow,dpow,mult,cat,e,f]=smodel_(s_input)
a=str2cell(s_input,'=');
i=1;
eqind=0; 
la =length(a);
while i<la
    if strcmp(a{i},'=')
        if strcmp(a{i+1},'=')
            i=i+1;
        else
            eqind=i;
            i=la;
        end
    end
    i=i+1;
end
if ~eqind
    if strcmp(a{la},'=')
        eqind = la;
    end
end
switch eqind
    case 0
        error('"=" is missing');
    case 1
        error('The response is missing');
    case la
        error('The model is missing');
end
y=''; 
for i=1:(eqind-1)
    y = [y a{i}];
end
b=''; 
for i=(eqind+1):la
    b = [b a{i}];
end
b=str2cell(b,'+-');
n=(length(b)+1)/2;
c=cell(1,n);
e=cell(1,n);
pos=ones(1,n);  
pow=zeros(1,n); 
cpow=zeros(1,n); 
dpow=zeros(1,n); 
mult=zeros(1,n); 
termsAll = '';
dolls=[];
k=1;
for i=1:n
    c{i}=str2cell(b{i*2-1},'*|@#');
    d{i}={};
    e{i}=[];
    if i>1 
        if strcmp(b{(i-1)*2},'-')
            pos(i)=0;
        end
    end
    j=1;
    if length(c{i})==1
        m=str2cell(c{i}{1},'^');
    else
        if length(str2cell(b{i*2-1},'^'))>1
            err_s = sprintf('"%s", Expression not allowed',b{i*2-1});
            error(err_s);
        end
        m=[];
    end
    if length(m)==3
        e{i} = [k];
        k=k+1;
        [s,doll]=sdoll(m{1});
        termsAll = strvcat(termsAll,s);
        dolls = [dolls doll];
        dpow(i) = str2num(m{3});
    else
        while j<=length(c{i})
            if  ~strcmp(c{i}{j},'@') & ~strcmp(c{i}{j},'#')
                if  ~strcmp(c{i}{j},'|') & ~strcmp(c{i}{j},'*')
                    e{i} = [e{i} k];
                    k=k+1;
                    [s,doll]=sdoll(c{i}{j});
                    termsAll = strvcat(termsAll,s);
                    dolls = [dolls doll];
                else
                    if  strcmp(c{i}{j},'*')
                        if length(str2cell(b{i*2-1},'|'))>1
                            err_s = sprintf('"%s", Expression not allowed.',b{i*2-1});
                            error(err_s);
                        end
                        mult(i)=1;
                    end
                end
            else
                if  strcmp(c{i}{j},'#')
                    j=j+1;
                    cpow(i)=str2num(c{i}{j});
                else
                    j=j+1;
                    pow(i)=str2num(c{i}{j});
                end
            end
            j=j+1;
        end
    end
end
[g,v]=distinct(termsAll);
g__=g;
for i=1:length(g)
    g__(i) = min(find(g==g(i)));
end
[g_,v__]=distinct(g__);
v_ = v;
for i=1:length(v__)
    v_(i,:) = termsAll(v__(i),:);
end
for i=1:n
    for j=1:length(e{i})
        e{i}(j) = g_(e{i}(j));
    end
end
nVar = size(v,1);
f=cell(1,nVar);
for i=1:nVar
    f{i}=deblank(strjust(v_(i,:),'left'));
end
cat =zeros(1,nVar); 
cat(g_(dolls==1))=1;
function  [s,doll]=sdoll(s)
doll=0;
if s(length(s))=='$'
    s=s(1:(length(s)-1));
    doll=1;
 end
function x=str2cell(s,symbols)
r=cell(1,length(symbols));
for i=1:length(symbols)
    r{i}=findstr(s,symbols(i));
end
x=cell(1,0);
m=1;
[r_m r_i ] =min_r(r);
while isfinite(r_m)
    if r_m>m
        st =deblank(strjust(s(m:(r_m-1)),'left'));
        if length(st)
            x = {x{:},st};
        end
    end
    x = {x{:},s(r_m)};
    r{r_i}=r{r_i}(2:end);
    m=r_m+1;
    [r_m r_i ] =min_r(r);
end
st =deblank(strjust(s(m:length(s)),'left'));
if length(st)
    x = {x{:},st};
end
function  [r_m,r_i] =min_r(r)
r_m = Inf;
r_i = -1;
for i=1:length(r)
    min_r  = min(r{i});
    if min_r <r_m;
        r_m = min_r;
        r_i = i;
    end
end
function [Xa,factors] = absStand(X,factors)
if(iscell(X))
    [Xa df] = c2m(X);
else
    Xa = X;
end
if(nargin<2)
    factors = max(abs(Xa)); 
end
for i=1:size(Xa,2) 
    if(factors(i)>0)
        Xa(:,i) = Xa(:,i)/factors(i); 
    else
        factors(i)=1;
    end
end 
if(iscell(X))
    Xa = m2c(Xa,df);
end
function Xadj = adjust(X,Y) 
orthY = myorth(Y);
orthX = myorth(X);
Xadj = X(:,[]);
rankXadj = size(myorth([orthX,orthY]),2) - size(orthY,2);
if(rankXadj==0)
   return;
end;
Xadj = myorth(orthX - orthY*(orthY'*orthX));
function df = c2df(C)
df = size(C{1},2);
for i=2:length(C)
   df = [df size(C{i},2)];
end
function [M,df] = c2m(C)
M = C{1};
df = size(C{1},2);
for i=2:length(C) 
   M = [M C{i}];
   df = [df size(C{i},2)];
end
function Yc = center(Y)
Yc = Y - ones(size(Y,1),1)*mean(Y);
function [group,distvalues] = distinct(X)
nRows = size(X,1);
[Y,index] = sortrows(X);
group = zeros(nRows,1);
group(index(1)) = 1;
distvalues = Y(1,:);
for i=2:nRows
   if(sum(Y(i,:)~=Y(i-1,:))==0)
      group(index(i)) = group(index(i-1));
   else
      group(index(i)) = group(index(i-1))+1;
      distvalues = [distvalues',Y(i,:)']';
   end
end      
function [dummy] = dummy_var(X)
[group,distvalues] = distinct(X);
dummy = (group==1);
for i=2:max(group) 
   dummy = [dummy,(group==i)];
end
dummy = double(dummy);
function [U,S,V] = economySVD(X)
[m,n] = size(X);
if(n>m)
    [V,S,U] = svd(X',0);
else
    [U,S,V] = svd(X,0);
end
function [exVar1,exVar2,dim,dimX,dimY,bufferDim,D,E,A,M,pD,pE,pA,pM] = ... 
ffmanovatest(modelData,errorData,part,partBufDim,minBufDim,maxBufDim,minErrDf,cp,stand)
if(iscell(errorData)) 
    dfError = errorData{2};
    errorData = errorData{1};
    nZeroRows = dfError-size(errorData,1);
    min_nZeroRows = size(errorData,2) - size(errorData,1);
    min_nZeroRows = min(min_nZeroRows,nZeroRows);
    if(min_nZeroRows>0)
        min_nZeroRows
        errorData = [errorData' zeros(size(errorData,2),min_nZeroRows)]';
    end    
else
    dfError = size(errorData,1);
end
Y    = [modelData',errorData']';
dimX = size(modelData,1);
dimFull = dfError + size(modelData,1); 
dimYfull = min(dimFull,size(Y,2));
maxDimY  = max(1,min(dimFull-dimX-minErrDf,dimYfull));
if(dimX==0 | dimYfull==0 | size(errorData,1)==0)
   exVar1=0;
   exVar2=0;
   dim=0;
   dimY=0;
   bufferDim=0;
   D=0;
   E=0;
   A=0;
   M=0;
   pD=99;
   pE=99;
   pA=99;
   pM=99;
   return;
end
if(stand)
  for i=1:size(Y,2)  
      Y(:,i) = Y(:,i)/(norm(Y(:,i))); 
  end 
end
if(size(Y,2)>size(Y,1))
    [U,S,V] = economySVD(Y);
else
    [U,S,V] = svd(Y); 
end
if(dimYfull==1 | dimFull==1)
   ss=S(1,1).^2;
else
   ss=diag(S).^2;
end
dimY  = 0;
part_ = 0;
part_dimY = 1;
while(part_<part_dimY)
   dimY = dimY+1;
   varMod  = sum(ss(1:dimY));
   varRest = sum(ss) - varMod;
   if(dimY==maxDimY)
      part_ = 1000;
   else
      factor = sum(((dimY+1):dimFull).^(cp)) /  sum(((dimY+1):dimYfull).^(cp)); 
      part_   = varMod / (varMod + factor*varRest);
   end
   if(length(part) >= dimY )
      part_dimY=part(dimY);
   end
end
bufferDim = max(0,floor(1.0001*min(dimYfull-dimY,...
   partBufDim*(dimFull-dimX-dimY-minErrDf))));
if(bufferDim>maxBufDim) 
   bufferDim = max(maxBufDim,0);
end
if(bufferDim<minBufDim)
   bufferDim = min(minBufDim,max(0,min(dimYfull-dimY,dimFull-dimX-dimY-minErrDf)));
end
exVar1 = varMod/sum(ss);
exVar2 = sum(ss(1:(dimY+bufferDim)))/sum(ss);
dim = dimFull - bufferDim;
XtY = (U(1:dimX,:))';
XtY = XtY([1:dimY,(dimY+bufferDim+1):end],:); 
[XtY,qrR] = qr(XtY,0);
XtY = XtY(1:dimY,:);
[U,S,V] = economySVD(XtY);
if(size(XtY,1)==1 | size(XtY,2)==1)
   ss=S(1,1).^2;
else
   ss=diag(S).^2;
end 
[D,E,A,M] = multiStatistics(ss);
[pD,pE,pA,pM] = multiPvalues(D,E,A,M,dim,dimX,dimY);
function [D,E,A,M] = multiStatistics(ss)
D = 1;
A = 0;
M = 0;
for i=1:length(ss)
   D = D*(1-ss(i));
   if( (1-ss(i))>0 )
      A = A+(ss(i)/(1-ss(i)));
   else
      A = A+ 1/eps;
   end
   if(i==1)
      E = A;
   end
   M = M+ss(i);
end
function [pD,pE,pA,pM] = multiPvalues(D,E,A,M,dim,dimX,dimY)
p = dimY;
q = dimX;
v = dim - dimX;
s = min(p,q);
m = (abs(p-q)-1)/2;
n = (v-p-1)/2;
r = v - (p-q+1)/2;
u = (p*q-2)/4;
t = 1;
if( (p^2+q^2-5)>0 )
   t = sqrt( (p^2*q^2-4)/(p^2+q^2-5) );
end
r_ = max(p,q); 
if( D^(1/t) <=0 )
   fD=1e+100;
else
   fD = ((1-D^(1/t))/D^(1/t))*((r*t-2*u)/(p*q));
end
pD = my_pValueF(fD,p*q,r*t-2*u); 
fE = E*(v-r_+q)/r_;
pE = my_pValueF(fE,r_,v-r_+q); 
fA = A*2*(s*n+1)/(s^2*(2*m+s+1));
pA = my_pValueF(fA,s*(2*m+s+1),2*(s*n+1));
if((s-M)<=0)
   fM=1e+100;
else
   fM = (M/(s-M))*( (2*n+s+1)/(2*m+s+1) );
end
pM = my_pValueF(fM,s*(2*m+s+1),s*(2*n+s+1));
function pValue = my_pValueF(f,ny1,ny2)
pValue = 100;
if(isreal(f) & f>0 & ny1>0.9 & ny2>0.9)
   if(f<1e-13)
      pValue = 1;
   else
      pValue = betainc(ny2/(ny2+ny1*f),ny2/2,ny1/2);
   end   
end
function estimable = is_estimable(Xnew,VextraDivS1)
estimable_lim = 1e-12; 
estimable = ( max(abs(Xnew*VextraDivS1),[],2) < estimable_lim);
if(isempty(estimable))
    estimable=ones(size(Xnew,1),1);
end
function [BetaU,msError,errorObs,Yhat] = linregEnd(Umodel,Y)
BetaU = Umodel'*Y;
Yhat = Umodel*BetaU;
[U S V] = economySVD(Y-Yhat);
df_error = size(Umodel,1) - size(Umodel,2);
errorObs = S*V'; 
if(size(errorObs,1)>df_error)
    errorObs = errorObs(1:df_error,:);
end
msError=sum(errorObs.*errorObs,1)/df_error;
if(size(errorObs,1)<df_error)
    errorObs = {errorObs,df_error};  
end
function [BetaU,VmodelDivS,VextraDivS1,msError,errorObs,Yhat] = linregEst(X,Y)
[Umodel,VmodelDivS,VextraDivS1] = linregStart(X);
[BetaU,msError,errorObs,Yhat] = linregEnd(Umodel,Y);
function [Umodel,VmodelDivS,VextraDivS1] = linregStart(X)
rank_lim = 1e-9; 
nObs = size(X,1);
[U,S,V] = svd(X,0);
S = diag(S);
r = length(S);
tol = max(size(U)) * S(1) * rank_lim; 
while S(r) < tol;
   r=r-1;
end
S=S(1:r); 
Umodel = U(:,1:r);
Vmodel = V(:,1:r);
VmodelDivS = Vmodel ./ (ones(size(V,1),1) * S');
Vextra = V(:,(r+1):size(V,2));
VextraDivS1 = Vextra/S(1);
function termNames = makeTermNames(names,model) 
ConstName      = 'Const';   
ErrorName      = 'Error';   
powerSymbol    = '^';       
multiplySymbol = '*';       
termNames = cell(1,size(model,1)+1);
for i=1:size(model,1) 
    if(sum(model(i,:))== 0)
        termNames{i} = ConstName;
    else
        s='';
        for j=1:size(model,2)
            mij = model(i,j);
            if(mij)
                if(~isempty(s))
                    s = [s multiplySymbol];
                end
                s = [s names{j}];
                if(mij>1)
                    s = [s sprintf('%s%d',powerSymbol,mij)];
                end
            end
        end
        termNames{i} = s;
    end
end
termNames{size(model,1)+1}=ErrorName;
function C = m2c(M,df)
if nargin < 2
    df = ones(1,size(M,2));
end
C=cell(1,length(df));
k=0;
for i=1:length(df) 
    C{i} = M(:,(k+1):(k+df(i)));
    k=k+df(i);
end
function results = manova_5050(xObj,Y,stand)
partBufDim = 0.5;        
minBufDim = 0;           
maxBufDim = 100000000;   
minErrDf  = 3;           
cp = -1;                 
part1 = 0.9;             
part2 = 0.5;             
part  = [part1,part2]';  
yNames=[];
if(stand)
   Y = stdStand(Y);
end  
model = xObj.model;
xyObj = xy_Obj(xObj,Y,yNames);
nTerms = length(xyObj.xObj.df_D_test);
results.termNames = xyObj.xObj.termNames;
results.exVarSS = xyObj.ss / xyObj.ssTot;
results.df = [xyObj.xObj.df_D_test xyObj.xObj.df_error];
results.df_om = [xyObj.xObj.df_D_om xyObj.xObj.df_error];
nPC = [];
nBU = [];
exVarPC = [];
exVarBU = [];
pValues = [];
normY = norm(Y);
for i=1:nTerms
   modelData = xyObj.hypObs{i};
   if(iscell(xyObj.errorObs))
       normTest =  norm([(xyObj.errorObs{1})',modelData']);
       dfError = xyObj.errorObs{2};
   else
       normTest =  norm([xyObj.errorObs',modelData']);
       dfError = size(xyObj.errorObs,1);
   end
   if(normY < 1e-250 | normTest/normY < 1e-12)
      [exVar1_,exVar2_,dim_,dimX_,dimY_,bufferDim_,D_,E_,A_,M_,pD_,pE_,pA_,pM_] = ...
         ffmanovatest(modelData(:,[]),zeros(dfError,0),part,partBufDim,minBufDim, ... 
         maxBufDim,minErrDf,cp,stand);
   else
      [exVar1_,exVar2_,dim_,dimX_,dimY_,bufferDim_,D_,E_,A_,M_,pD_,pE_,pA_,pM_] = ... 
         ffmanovatest(modelData,xyObj.errorObs,part,partBufDim,minBufDim,...
         maxBufDim,minErrDf,cp,stand);
   end
   nPC = [nPC dimY_];
   nBU = [nBU bufferDim_];
   exVarPC = [exVarPC exVar1_];
   exVarBU = [exVarBU exVar2_];
   pValues = [pValues pA_];
end
results.nPC = nPC;
results.nBU = nBU;
results.exVarPC = exVarPC;
results.exVarBU = exVarBU;
results.pValues = pValues;
    outputText=[];
    outputText=outLine(outputText,sprintf('  --- 50-50 MANOVA Version 2.0 --- %d objects -- %d responses:',size(Y,1),size(Y,2)));
    approx = 0;
    names = '';
    for i=1:length(results.termNames)
        names = strvcat(names,results.termNames{i});
    end
    names = strvcat(names,'Source');
    outputText=outLine(outputText,sprintf('  %s  DF        exVarSS nPC nBu exVarPC exVarBU    p-Value ',names(size(model,1)+2,:)));
    for i=1:(nTerms+1)
       s1 = sprintf('  %s',names(i,:));
       s2 = sprintf('%4d',results.df(i));
       if(results.df(i)==results.df_om(i))
          dfFull = '       ';
      else
          dfFull = sprintf('(%d)',results.df_om(i));
      end
       dfFull = strjust(sprintf('%7s',dfFull),'left');
       s3 = sprintf('%s',dfFull(1:5));
       s4 = sprintf('  %8.6f',results.exVarSS(i));
       if(i <= nTerms)
          s5 = sprintf(' %3d',nPC(i));
          s6 = sprintf(' %3d ',nBU(i));
          s7 = sprintf(' %5.3f  ',exVarPC(i));
          s8 = sprintf(' %5.3f    ',exVarBU(i));
          if(pValues(i)<2)
             s9 = sprintf('%8.6f ',pValues(i));
             if(nPC(i)>2 & results.df(i)>2)
                s10 =sprintf('x');
                approx = 1;
             else
                s10 = ' ';
             end
          else
             s9 =sprintf(' ....... ');
             s10 = ' ';
          end
          outputText=outLine(outputText,sprintf('%s%s%s%s%s%s%s%s%s%s',s1,s2,s3,s4,s5,s6,s7,s8,s9,s10));
       end  
    end
    if(stand)
       s5 = sprintf(' - STANDARDIZATION ON  ');
    else 
       s5 = sprintf(' - STANDARDIZATION OFF ');
    end
    if(approx)
       s6 = sprintf('- x Approx p');
    else
       s6 = sprintf('------------');
    end
    outputText=outLine(outputText,sprintf('%s%s%s%s%s%s%s%s%s%s',s1,s2,s3,s4,s5,s6));
results.outputText = outputText;
function outputText=outLine(text,line)
outputText=strvcat(text,line);
function [results] = manova5050nostring(X,Y,cova,model,xNames,stand,nSim,Xnew,cXnew,nSimXNew)
if isempty(X)
    if nargin < 3
        nXvar = 0;
    else
        nXvar = length(cova);
    end
    nObs = 0;
else
    if ~iscell(X)
        X = m2c(X);
    end
    nXvar = size(X,2);
    nObs = size(X,1);
end
if nargin < 7
   nSim=[];
end
if nargin < 6
   stand=[];
end
if nargin < 5 
    xNames=[];
end
if nargin < 4 
    model=[];
end
if nargin < 3
   cova=[];
end
if nargin < 2
   Y=[]; 
end
if isempty(nSim)
   nSim=-1;
end
if isempty(stand)
   stand=1;
end
if isempty(xNames)
   xNames = cell(1,nXvar);
   for i=1:nXvar 
      if(i<24) 
         xNames{i} = sprintf('%c','A'-1+i);
      else
         xNames{i}=sprintf('x%d',i);
     end    
   end
end
if isempty(model)
   model=eye(nXvar);
end
if isempty(cova)
   cova=zeros(1,nXvar);
end
if isempty(X) | isempty(Y)
        results = modelmatrix(cova,model,xNames,0);
    return;
end
if iscell(model)|ischar(model)
    model = modelmatrix(cova,model,xNames,1);
end
if(max(size(nSim))==1)
    nSim = nSim*ones(1,size(model,1));
end
model = [(zeros(size(model(1,:))))' model']';
if(size(nSim,1)>size(nSim,2))
    nSim = nSim';
end
nSim = [-1 nSim];  
nTerms = size(model,1);
obj = x_Obj(X,cova,model,xNames);
results = manova_5050(obj,Y,stand);
results.termNames = results.termNames(2:(nTerms+1));
results.exVarSS = results.exVarSS(2:(nTerms+1));
results.df = results.df(2:(nTerms+1));
results.df_om = results.df_om(2:(nTerms+1));
results.nPC = results.nPC(2:nTerms);
results.nBU = results.nBU(2:nTerms);
results.exVarPC = results.exVarPC(2:nTerms);
results.exVarBU = results.exVarBU(2:nTerms);
results.pValues = results.pValues(2:nTerms);
disp_rows = 1:size(results.outputText,1);
results.outputText = results.outputText(disp_rows~=3,:);
disp(' ');
disp(results.outputText);  
disp(' ');
obj = xy_Obj(obj,Y,[]);
results.nSim = nSim(2:nTerms);
if(max(nSim)>=0) 
    results.pRaw = nan * ones(size(model,1)-1,size(Y,2));
    results.stat = results.pRaw;
    for i=2:size(model,1)   
       if(nSim(i)>=0)
          if(iscell(obj.errorObs)) 
              [results.pRaw(i-1,:),results.stat(i-1,:)] = uniTest(obj.hypObs{i},obj.errorObs{1},obj.errorObs{2});
          else
              [results.pRaw(i-1,:),results.stat(i-1,:)] = uniTest(obj.hypObs{i},obj.errorObs);
          end
       else
          results.nSim(i-1) = 0;
       end
    end
else
    results.pRaw = [];
    results.stat = [];
    results.nSim = zeros(size(results.nSim));
end
if(max(nSim)>0) 
    results.pAdjusted = nan * ones(size(model,1)-1,size(Y,2));
    results.pAdjFDR = results.pAdjusted;
    for i=2:size(model,1)    
       if(nSim(i)>0)
          fprintf('   %20s ...',obj.xObj.termNames{i});
          if(iscell(obj.errorObs)) 
              [pAdjusted,pAdjFDR,nSim_] = rotationtest(obj.hypObs{i},obj.errorObs{1},nSim(i),obj.errorObs{2});  
          else
              [pAdjusted,pAdjFDR,nSim_] = rotationtest(obj.hypObs{i},obj.errorObs,nSim(i));  
          end
           results.pAdjusted(i-1,:) = pAdjusted;
           results.pAdjFDR(i-1,:) = pAdjFDR;
           results.nSim(i-1) = nSim_;
           fprintf('  %d simulations performed \n',nSim_);
       end
    end
else
    results.pAdjusted = [];
    results.pAdjFDR = [];
end
results.Yhat = obj.Yhat;
results.YhatStd = obj.YhatStd;
if nargin > 7
    if size(Xnew,2)>1 | isnumeric(Xnew)
        Xnew = {Xnew};
    end
    nXnew = size(Xnew,1);
    results.newPred = cell(nXnew,1);
    for i=1:nXnew
        Xnew_i = Xnew{i};
        if ~iscell(Xnew_i)
            Xnew_i = m2c(Xnew_i);
        end
        switch nargin
            case 8
                [YnewPred,YnewStd,estimable] = pred(obj,Xnew_i);
                YnewPred_ = nan*ones(size(YnewPred));
                YnewStd_ = YnewPred_;
                YnewPred_(estimable==1,:) = YnewPred(estimable==1,:);
                YnewStd_(estimable==1,:) = YnewStd(estimable==1,:);
                results.newPred{i}.Yhat    = YnewPred_;
                results.newPred{i}.YhatStd = YnewStd_;
            case 9     
            case 10    
        end
    end
end
function m=modelmatrix(cova,model,names,noprint)
if ischar(model)
    switch model
        case 'linear',
            model = m2c(ones(size(cova)));
        case 'quadratic',
            model = m2c(2*ones(size(cova)));
        case 'cubic',
            model = m2c(3*ones(size(cova)));
    end
end
if iscell(model) 
    order=c2m(model);
    nVar=length(cova);
    if nargin < 4
        noprint = 0;
    end
    if nargin < 3 | isempty(names)
        names = cell(1,nVar);
        for i=1:nVar
            if(i<24)
                names{i} = sprintf('%c','A'-1+i);
            else
                names{i}=sprintf('x%d',i);
            end
        end
    end
    if(length(order)~=nVar)
        order_ = order;
        order = ones(size(cova));
        order_ = [order_ order*order_(1)];
        for i=1:nVar 
            order(i) = order_(i);
        end
    end
    m=[];
    levels=cell(1,max(order)+1);
    levels{1} = zeros(nVar,1);;
    for k=1:max(order)
        z = levels{k};
        y = [];
        for j=1:size(z,2) 
            ord=order(z(:,j)>0);
            ord=[Inf,ord];
            if(min(ord)>=k)
                for i=1:nVar 
                    if(order(i)>=k)
                        x = z(:,j);
                        x(i) = x(i)+1;
                        y = [y x];
                    end
                end
            end
        end
        [group,y] = distinct(y');
        y=y(end:(-1):1,:);
        sorty = -sort(-y,2);
        sizey2 = size(y,2);
        y = sortrows([sorty,y],1:sizey2);
        y = (y(:,(sizey2+1):end))';
        levels{k+1}=y;
        m = [m y];
    end
    m = m';
    for i=1:nVar 
        if(cova(i)==0)
            rows = m(:,i)<=1;
            m=m(rows,:);
        end
    end
else 
    m = model;
end
if ~noprint
    term_names = makeTermNames(names,m);
    for i=1:size(m,1) 
        tn =term_names{i};
        switch i
            case 1,
                fprintf('m = [ %s ; ... %% %4d: %s\n',num2str(m(i,:)),i,tn);
            case size(m,1),
                fprintf('      %s ];    %% %4d: %s\n',num2str(m(i,:)),i,tn);
            otherwise
                fprintf('      %s ; ... %% %4d: %s\n',num2str(m(i,:)),i,tn)
        end
    end
end
function [G,GN]=my_grp2idx(S)
if(isnumeric(S))
    [G,GN]=distinct(S);
else
    [G,GN]=grp2idx(S);
end
function D = my_x2fx(X,model) 
D = cell(1,size(model,1));
for i=1:size(model,1)  
    d = ones(size(X{1,1},1),1);
    for j=1:size(model,2) 
        for k=1:model(i,j)
            d = mult(d,X{j});
        end
    end
    D{i} = d;
end
function ab = mult(a,b);
ab = zeros(size(a,1),size(a,2)*size(b,2));
k=0;
for i=1:size(a,2) 
    for j=1:size(b,2) 
        k=k+1;
        ab(:,k) = a(:,i).*b(:,j);
    end
end
function U = myorth(X)
tol_ = 1e-9;   
if size(X,2)==0
   U=X; 
   return;
end
[U,S,V] = economySVD(X);
S = diag(S);
r = length(S);
meanS = mean(S);
tol = max(size(U)) * S(1) * tol_; 
while S(r) < tol;
   r=r-1;
end
U=U(:,1:r);
if( size(U,2)==1 & size(X,2)==1 ) 
    if( (X'*U) <0)                
        U = -U;
    end
end
function [Xa,means,stds] = normalize(X,means,stds)
if(iscell(X))
    [Xa df] = c2m(X);
else
    Xa = X;
end
if(nargin<2)
    stds = std(Xa);
    means = mean(Xa);
end
for i=1:size(Xa,2)
    if(stds(i)==0)    
        stds(i)=1;
        if(means(i)>0)
            means(i) = means(i)-1;
        end
    end
    Xa(:,i) = (Xa(:,i)-means(i))/stds(i); 
end 
if(iscell(X))
    Xa = m2c(Xa,df);
end
function [YnewPred,YnewStd,estimable,hypObs] = pred(xyObj,Xnew,c)
nVar = size(Xnew,2);
empX=zeros(1,nVar);
for i=1:nVar 
    if(isempty(Xnew{i}))
        empX(i)=1;
    else
        nObs = size(Xnew{i},1);
    end
end
X = Xnew;
for i=1:nVar 
    if(empX(i))
        X{i} = ones(nObs,1)*mean(xyObj.xObj.X{i},1);  
    else
        if(xyObj.xObj.cova(i)==0)
            [G_,GN_]=my_grp2idx(Xnew{i});
            Xnew_i = GN_(G_);  
            GN=xyObj.xObj.catNames{i};
            df = length(GN);
            x = zeros(nObs,df);
            for j=1:df 
               for k=1:nObs 
                 if(lik(Xnew_i(k,:),GN(j,:)))
                    x(k,j)=1;
                  end
                end
            end
            X{i} = x;
        end
    end
end
X_norm = normalize(X,xyObj.xObj.X_norm_means,xyObj.xObj.X_norm_stds);
Dnew = my_x2fx(X_norm,xyObj.xObj.model);
Dnew_ = c2m(Dnew);
if(nargin > 2)
    Dnew_ = c*Dnew_;
end
estimable_D = is_estimable(Dnew_,xyObj.xObj.VextraDivS1_D);
Unew_D = Dnew_ * xyObj.xObj.VmodelDivS_D;
D_om_new_ = Unew_D*xyObj.xObj.Beta_D;
D_om_new = m2c(D_om_new_,xyObj.xObj.df_D_om);
emp = empX*xyObj.xObj.model'>0;
for i=1:length(Dnew) 
    if(emp(i))
        D_om_new{i} = ones(nObs,1)*mean(xyObj.xObj.D_om{i},1);
        if(nargin > 2)
            D_om_new{i} = c* D_om_new{i}; 
        end
    end
end
D_om_new_ = c2m(D_om_new);
estimable = is_estimable(D_om_new_,xyObj.xObj.VextraDivS1);
Unew = D_om_new_ * xyObj.xObj.VmodelDivS;
YnewPred = Unew*xyObj.Beta;
YnewStd = sqrt(sum(Unew .^ 2,2)*xyObj.msError);
estimable = estimable .* estimable_D;
if(nargin>2 & nargout>3)
    hypObs = (myorth(xyObj.xObj.Umodel*Unew'))'*xyObj.Y;
end
function c=lik(a,b)
if(isnumeric(a))
    c=min(a==b);
else
    c=strcmp(a,b);
end
function [pAdjusted,pAdjFDR,simN] = ... 
rotationtest(modelData,errorData,simN,dfE)
dispsim_default = 1;  
dispsim=dispsim_default;
dfH = size(modelData,1);
if(nargin<4)  
    dfE = size(errorData,1);
end
Y    = [modelData',errorData']';
q = size(Y,2);
dfT = dfH + dfE; 
dfT_ = size(Y,1);
X = zeros(dfT,dfH); 
X(1:dfH,1:dfH) = diag(ones(dfH,1));
if(dfH==0| dfE==0 | q==0)    
   pAdjusted=NaN*ones(1,q);
   pAdjFDR=NaN*ones(1,q);
   return;
end
ss =  zeros(1,q);
sss = zeros(1,q);
for j=1:q  
    if(norm(Y(:,j))>0) 
        Y(:,j) = Y(:,j)/(norm(Y(:,j))); 
    end 
    ss(j)= Y(1:dfH,j)' * Y(1:dfH,j);
end
Ys = sortrows([Y' ss' (1:q)'],dfT_+1);
sortindex = Ys(:,dfT_+2)';
ss = Ys(:,dfT_+1)';
Ys = (Ys(:,1:dfT_))';
sizeX=size(X);
m=ones(1,q);
mFDR=ones(1,q);
pAdjusted=ones(1,q);
pAdjFDR=ones(1,q);
if(dispsim)
    try
        infofig=siminfo('start');
        step=1;
        stepflag=10;
        t0=clock;
        simNinput=simN;
    catch
        dispsim=0;
    end
end
repsim=0;
if((dfT/dfT_)>10)  repsim=10; end   
if((dfT/dfT_)>100) repsim=100; end  
repindex = 0;
i=0;
while(i<simN)
    i=i+1;
    if(repsim)
        if(repindex==0)
            [Xs,r]=qr(randn(sizeX),0);
        end
        Z = Xs((repindex*dfT_+1):((repindex+1)*dfT_),:)' * Ys;
        repindex = mod(repindex+1,repsim);
    else
        [Xs,r]=qr(randn(sizeX),0);
        Z = Xs(1:dfT_,:)' * Ys;     
    end
    sss=sum(Z.*Z,1); 
    maxss=0;
    for j=1:q  
        maxss=max(maxss,sss(j));
        if(maxss>=ss(j)) m(j) = m(j)+1; end
    end    
    sss_sorted = [sort(sss) Inf];
    jObs=1;
    for j=1:q 
        while(sss_sorted(jObs) < ss(j))
            jObs=jObs+1;
        end
        mFDR(j) = mFDR(j) + min(1,(q+1-jObs)/(q+1-j));  
    end
    if(dispsim)
    if(mod(i,step)==0)
        if(i==10*step & stepflag==10)
            difftime=etime(clock,t0);
            difftime=(9/10)*difftime;
            if(difftime<2)
                if((10*step)<=simN/20) 
                    step=step*10; 
                end
            else
                if(difftime<5)
                    if((5*step)<=simN/20) 
                        step=step*5; 
                        stepflag=5;
                    end
                else
                    if(difftime<10)
                        if((2*step)<=simN/20) 
                            step=step*2; 
                            stepflag=2;
                        end
                    end
                end
            end
            t0=clock;
        end
        pause(0.001);  
        try 
            pause_=get(infofig,'UserData');
            if(pause_)
                uiwait(infofig);
            end   
            set(infofig,'UserData',0);
            set(findobj('Tag','text2'),'String',strvcat(' ',...
                sprintf('%d out of %d',i,simN),...
                sprintf('%5.2f%%',100*i/simN),' ',...
                sprintf('%d times',m(q)-1),' ',...
                sprintf('%8.6f',m(q)/(i+1))));
        catch
            simN = i;
        end
    end
    end
end
if(dispsim)
    try 
        delete(infofig);
    catch
    end
    if(simNinput~=simN)
        fprintf('  ***** Simulations stopped interactively *****');
    end
end
for j=1:q 
    pAdjusted(j) = m(j)/(simN+1);
end
for j=2:q 
    pAdjusted(q+1-j) = max(pAdjusted((q+1-j):(q+2-j)));
end
pAdjusted = sortrows([pAdjusted' sortindex'],2);
pAdjusted = (pAdjusted(:,1))';
pAdjFDR=ones(1,q);
for j=1:q 
    pAdjFDR(j) = mFDR(j)/(simN+1);
end
for j=2:q 
    pAdjFDR(j) = min(pAdjFDR((j-1):j)); 
end
pAdjFDR = sortrows([pAdjFDR' sortindex'],2);
pAdjFDR = (pAdjFDR(:,1))';
function output=siminfo(action,text)
switch(action)
case 'start'   
h0 = figure('Color',[0.752941176470588 0.752941176470588 0.752941176470588],...
	'Position',[200 200 380 300],'WindowStyle','modal',...
   'Tag','FigSiminfo','NumberTitle','off','Name','Rotation simulations');
h1 = uicontrol('Parent',h0, ...
	'Units','points', ...
	'Callback','ffmanova(''stop'');', ...
	'FontSize',12, ...
	'ListboxTop',0, ...
	'Position',[190 175 65 30], ...
	'String','STOP', ...
	'Tag','stop');
h1 = uicontrol('Parent',h0, ...
	'Units','points', ...
	'Callback','ffmanova(''pause'');', ...
	'FontSize',12, ...
	'ListboxTop',0, ...
	'Position',[105 175 65 30], ...
	'String','PAUSE', ...
	'Tag','pause');
h1 = uicontrol('Parent',h0, ...
	'Units','points', ...
	'Callback','ffmanova(''continue'');', ...
	'FontSize',12, ...
	'ListboxTop',0, ...
	'Position',[20 175 65 30], ...
	'String','CONTINUE', ...
    'Enable','off', ...
	'Tag','continue');
h1 = uicontrol('Parent',h0, ...
	'Units','points', ...
	'BackgroundColor',[1,1,1],...
	'HorizontalAlignment','right', ...
	'ListboxTop',0, ...
	'Position',[30 55 100 100], ...
    'FontSize',10, ...
	'Style','text', ...
    'String',...
    strvcat(' ','   Simulations: ',' ',' ','Fmax exceeded : ',' ','Minimum p-value = '), ...
	'Tag','text1');
h1 = uicontrol('Parent',h0, ...
	'Units','points', ...
	'BackgroundColor',[1,1,1],...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[130 55 120 100], ...
    'FontSize',10, ...
	'Style','text', ...
    'String','tull \n hei', ...
	'Tag','text2');
h1 = uicontrol('Parent',h0, ...
	'Units','points', ...
	'BackgroundColor',[1 1 0], ...
	'HorizontalAlignment','center', ...
	'ListboxTop',0, ...
	'Position',[30 30 220 23], ...
    'FontSize',18, ...
	'Style','text', ...
    'Visible','off',...
    'String', 'Program paused',...
	'Tag','pausetext');
h1 = uicontrol('Parent',h0, ...
	'Units','points', ...
	'HorizontalAlignment','center', ...
	'ListboxTop',0, ...
	'Position',[30 5 220 20], ...
    'FontSize',8, ...
	'Style','text', ...
    'String', strvcat('Øyvind Langsrud - MATFORSK - Copyright 2005',...
    'www.matforsk.no/ola'),...
	'Tag','authortext');
set(h0,'UserData',0);
output=h0;
case 'pause'
    set(gcf,'UserData',1);
    set(findobj('Tag','continue'),'Enable','on');
    set(findobj('Tag','pause'),'Enable','off');
    set(findobj('Tag','pausetext'),'Visible','on');
case 'continue'
    set(findobj('Tag','continue'),'Enable','off');
    set(findobj('Tag','pause'),'Enable','on');
    set(findobj('Tag','pausetext'),'Visible','off');
    uiresume(gcf);
case 'stop'
    delete(gcf);
end
function Ys = stdStand(Y)
Ys = Y;
stdY = std(Y);
for i=1:size(Y,2) 
    if(stdY(i)>0)
        Ys(:,i) = Y(:,i)/stdY(i); 
    end
end 
function [pValues,stat] = uniTest(modelData,errorData,dfError)
dfModel = size(modelData,1);
if(nargin<3)  
    dfError = size(errorData,1);
end
if(dfModel==0 | dfError==0)
   pValues=ones(1,size(modelData,2));
   stat=zeros(1,size(modelData,2));
   return;
end
errorSS = sum(errorData.^2,1);
if(dfModel==1) 
    stat = modelData ./  sqrt(errorSS/dfError);
    Fstat = stat.^2;
else 
    modelSS = sum(modelData.^2,1);
    stat=(dfError/dfModel) * (modelSS ./ errorSS);
    Fstat = stat;
end
pValues =my_pValueF_(Fstat,dfModel,dfError);
function pValue = my_pValueF_(f,ny1,ny2)
     pValue = betainc(ny2*((ny2+ny1*f).^(-1)),ny2/2,ny1/2);
function xObj = x_Obj(Xinput,cova,model,names)
xObj.Xinput = Xinput;
xObj.cova = cova;
xObj.model = model;
xObj.names = names;
xObj.termNames = makeTermNames(names,model);
X=Xinput;
nVar = size(X,2);
catNames = cell(size(X));
for i=1:nVar 
    if(cova(i)==0)
        [G,GN] = my_grp2idx(X{i});
        X{i}=dummy_var(G); 
        catNames{i}=GN;
    end
end
if(min(sum(model'))>0) 
    [X_norm X_norm_stds] = absStand(X);
    X_norm_means = zeros(size(X_norm_stds));
else
    [X_norm X_norm_means X_norm_stds] = normalize(X);
end
D = my_x2fx(X_norm,model);
D_om   = orth_D(D,model,'om');
D_test = orth_D(D_om,model,'test');  
[D_om_,df_D_om] = c2m(D_om);
df_D_test = c2df(D_test);
[Beta_D,VmodelDivS_D,VextraDivS1_D] = linregEst(c2m(D),D_om_);
[Umodel,VmodelDivS,VextraDivS1] = linregStart(D_om_);
xObj.df_error = size(Umodel,1) - size(Umodel,2);
xObj.nVar = nVar;
xObj.catNames = catNames;
xObj.X = X;
xObj.X_norm_means = X_norm_means;
xObj.X_norm_stds = X_norm_stds;
xObj.D = D;
xObj.D_test = D_test;
xObj.D_om = D_om;
xObj.df_D_om = df_D_om;
xObj.df_D_test = df_D_test;
xObj.Beta_D = Beta_D;
xObj.VmodelDivS_D = VmodelDivS_D;
xObj.VextraDivS1_D = VextraDivS1_D;
xObj.Umodel = Umodel;
xObj.VmodelDivS = VmodelDivS;
xObj.VextraDivS1 = VextraDivS1;
function Dorth = orth_D(D,model,method) 
Dorth = cell(1,size(model,1));
if(length(D)~=size(model,1))
    return;
end
for i=1:size(model,1)
    d = D{i};
    d_adj = d(:,[]);
    for j=1:size(model,1)
        switch lower(method)
            case {'test'} 
                if(min( model(j,:) - model(i,:)) < 0 )    
                    d_adj = [d_adj D{j}];
                end
            case {'om'}
                if( (min( model(j,:) - model(i,:)) < 0)  &  (max( model(j,:) - model(i,:)) <= 0) )    
                    d_adj = [d_adj D{j}];
                end
            case {'seq'}
                if(j<i)
                    d_adj = [d_adj D{j}];
                end
            case {'ssIII'}
                if(j~=i)
                    d_adj = [d_adj D{j}];
                end
        end
    end
    Dorth{i} =  adjust(d,d_adj);
end
function xyObj = xy_Obj(xObj,Y,yNames)
xyObj.xObj=xObj;
xyObj.Y=Y;
xyObj.yNames=yNames;
[xyObj.Beta,xyObj.msError,xyObj.errorObs,xyObj.Yhat] = linregEnd(xObj.Umodel,Y); 
ss=[];
xyObj.YhatStd = sqrt(sum(xObj.Umodel.^ 2,2)*xyObj.msError);
hypObs = cell(size(xObj.D_test));
for i=1:length(xObj.D_test)
    hObs = xObj.D_test{i}'*Y;
    hypObs{i} = hObs;
    ss = [ss sum(sum(hObs.^2))];
end
if(iscell(xyObj.errorObs)) 
    ss = [ss xyObj.errorObs{2}*sum(xyObj.msError)];
else
    ss = [ss size(xyObj.errorObs,1)*sum(xyObj.msError)];
end
xyObj.ssTotFull = sum(sum(Y.^2));
xyObj.ssTot     = sum(sum(center(Y).^2));
xyObj.ss = ss;
xyObj.hypObs = hypObs;
