function [error, Ypred, YpredAbsErrBlock, SOs] = SO_PLS_crossval(X,Y,ncomps, seg_n,seg_type, varargin)
%% Globally optimized SO-PLS
%
% MANDATORY ARGUMENTS
% X           - cells containing predictors
% Y           - matrix containing response(s)
% ncomps      - number of component per block
%               (single integer means maximum total number of components)
%
% seg_n       - number of cross-validation segments
% seg_type    - cross-validation segment type (see cvseg function)
%
% OPTIONAL ARGUMENTS
% class       - 0/1 vector marking responses as categorical
%               (coded 1,2,... for classification)
% LDA         - 1 if classification by LDA instead of max. pred.
% standardize - standardize all continuous responses
% Yadd        - additional response information (fixed)
% lowers      - lower gamma value(s) in CPPLS (per block)
% uppers      - upper --------------- || ----------------
% weights     - object weights in CPPLS
% restricted  - calculate scores restricted to current block's row-space
%
% SVD         - number of SVD scores from other blocks used as
%               additional responses (per block). Negative numbers
%               gives fitted responses instead of scores.
% PLS         - number of PLS scores ... (see SVD)
%
%
% OUTPUT
% error       - root mean squared error of cross-validation for contiuous
%               responses, proportion of mis-classification for categorical
% Ypred       - cross-validated predictions of responses
% YpredAbsErrBlock - absolute error of cummulative predictions per block
% SOs         - cells containing SO structs for each cross-validation
%               segment

% Extra arguments and defaults
names = {'standardize' 'Yadd' 'lowers' 'uppers' 'weights' 'SVD' 'PLS' 'class' 'conf' 'LDA' 'restricted'};
dflts = {           0     []      0.5      0.5        []     0     0      []       0     0       false};
[standardize, Yadd,lowers,uppers,weights,SVD,PLS,class,conf,LDA,restricted] = match_arguments(names,dflts,varargin{:});

[n,q] = size(Y);
if seg_type == 5
    cv = seg_n; % Custom CV segments
    seg_n = max(cv);
else
    [cv, ~] = cvseg(n,seg_n,seg_type);
end

if isempty(class)
    class = zeros(1,q);
end

% Construct Y for CPPLS based on possible categorical responses
Y_orig = Y;
Y_wide = [];
dummy  = [];
Ywidth = zeros(q,1);
for i=1:q
    if class(i)==1
        Yd = dummyfy(Y(:,i));
        Y_wide = [Y_wide Yd];
        Ywidth(i) = size(Yd,2);
        dummy = [dummy ones(1,Ywidth(i))];
    else
        Y_wide = [Y_wide Y(:,i)];
        Ywidth(i) = 1;
        dummy = [dummy 0];
    end
end
Y = Y_wide;

% Initialize variables
nb     = length(X);
if length(ncomps) == 1
    maxcomp = min(n-1, ncomps);
    ncol    = @(x)size(x,2);
    ncomps  = min(cellfun(ncol,X), maxcomp)+1;
%     ncomps  = repmat(ncomps+1,1,nb);
else
    maxcomp = [];
    ncomps  = ncomps + 1; % Compensate for zero component
end
nc     = prod(ncomps);
error  = zeros([ncomps q]);
Ypred  = cell(ncomps); for i=1:nc; Ypred{i} = zeros(n,sum(Ywidth));end
Ypred_post = cell(ncomps); for i=1:nc; Ypred_post{i} = zeros(n,q);end
YpredAbsErrBlock = cell(ncomps); for i=1:nc; YpredAbsErrBlock{i} = zeros(n,sum(Ywidth),nb+1);end
if ~isempty(maxcomp)
    for i = 1:nc
        dims = ind2subS(ncomps, i);
        if sum(dims-1) > maxcomp
            Ypred{i} = [];
            Ypred_post{i} = [];
            YpredAbsErrBlock{i} = [];
        end
    end
    for i = 1:(nc*q)
        dims = ind2subS([ncomps q], i);
        if sum(dims-1) > maxcomp
            error(i) = NaN;
        end
    end
end
SOs    = cell(seg_n,1);
DynYadd = [];

% Prepare for classification by LDA
if LDA == 1
    Y_class = cell(ncomps); for i=1:nc; Y_class{i} = zeros(n,1);end
%     Y_class = zeros(n,nc);
end

% Extend PLS/SVD if common component number supplied
if SVD(1)>0 && length(SVD)==1
    SVD = repmat(SVD,1,nb);
end
if PLS(1)>0 && length(PLS)==1
    PLS = repmat(PLS,1,nb);
end

% Standardize
if standardize == 1
    sd = std(Y(:,dummy==0));
    me = mean(Y(:,dummy==0));
    Y(:,dummy==0)  = (Y(:,dummy==0) - repmat(me,n,1))./repmat(sd,n,1);
end

% Main cross-validation loop
fprintf('\nSegments computed\n 0%%');
for a = 1:seg_n
    % Segmentation
    [Xin, Xout] = cellsub(X,nb,cv,a);
    Yin  = Y(cv~=a,:);
    if ~isempty(Yadd) % Pre-made Yadd
        Yaddin = Yadd(cv~=a,:);
    else
        Yaddin = [];
    end
    
    % Dynamic Yadd
    if SVD(1) ~= 0
        DynYadd = cell(1,nb);
        for i = 1:nb
            [DynYadd{1,i},~,~] = svds(Xin{i},abs(SVD(i)));
        end
        if SVD(1) < 0
            for i = 1:nb
                DynYadd{1,i} = DynYadd{1,i} * (DynYadd{1,i}\Yin);
            end
        end
    end
    if PLS(1) ~= 0
        DynYadd = cell(1,nb);
        for i = 1:nb
            [~, ~, DynYadd{1,i}] = cppls(abs(PLS(i)), Xin{i}, Yin);
        end
        if PLS(1) < 0 % Y_hat instead of scores:
            for i = 1:nb
                [~, ~, ~, ~,~, beta] = cppls(abs(PLS(i)), Xin{i}, Yin);
                DynYadd{1,i} = Xin{i}*beta;
            end
        end
    end
    
    % Modeling
    if ~isempty(maxcomp)
%         ncomp = ncomps(1)-1;
        ncomp = ncomps - 1;
    else
        ncomp = ncomps - 1;
    end
    SO = SO_PLS_rec(Xin,Yin,ncomp,restricted, Yaddin, DynYadd, lowers,uppers,weights,0,conf);
    
    % Prediction
    for i = 1:nc
        dims = ind2subS(ncomps, i);
        compute = 1;
        if ~isempty(maxcomp) && sum(dims-1) > maxcomp
            compute = 0;
        end
        if compute == 1
            SOut = SO_PLS_from_rec(SO, dims-1);
            [Ypred_ai, YpredBlock_ai] = SO_PLS_pred(SOut, Xout);
            %         Ypred_ai(:,dummy==1) = (Ypred_ai(:,dummy==1)>0.5)*1;
            Ypred = cellinsert(Ypred, dims, cv, a, Ypred_ai);
            if nargout > 2
                YpredAbsErrBlock = cellinsert3(YpredAbsErrBlock, dims, cv, a, YpredBlock_ai);
            end
            if LDA == 1
                [Y_class_ai] = SO_PLS_class(SOut, Xout);
                %         Ypred_ai(:,dummy==1) = (Ypred_ai(:,dummy==1)>0.5)*1;
                Y_class = cellinsert(Y_class, dims, cv, a, Y_class_ai);
            end
        end
    end
    if nargout == 4
        SOs{a,1} = SO;
    end
    
    % Progress indication
    percent = round(100*a/seg_n);
    last = round(100*(a-1)/seg_n);
    if last <10
        fprintf('\b\b%d%%',percent)
    elseif last <100
        fprintf('\b\b\b%d%%',percent)
    else
        fprintf('\b\b\b\b%d%%',percent)
    end
end

for i = 1:nc
    compute = 1;
    dims = ind2subS(ncomps, i);
    if ~isempty(maxcomp) && sum(dims-1) > maxcomp
        compute = 0;
    end
    if compute == 1 && LDA ~= 1
        if standardize == 1
            Ypred{i}(:,dummy==0) = Ypred{i}(:,dummy==0).*repmat(sd,n,1) + repmat(me,n,1);
            if nargout > 2
                YpredAbsErrBlock{i}(:,dummy==0,:) = cumsum(YpredAbsErrBlock{i}(:,dummy==0,:),3).*repmat(sd,[n,1,nb+1]) + repmat(me,[n,1,nb+1]);
                YpredAbsErrBlock{i}(:,dummy==1,:) = cumsum(YpredAbsErrBlock{i}(:,dummy==1,:),3);
                YpredAbsErrBlock{i} = abs(YpredAbsErrBlock{i} - repmat(Y_wide,[1,1,(nb+1)]));
            end
        else
            if nargout > 2
                YpredAbsErrBlock{i} = abs(cumsum(YpredAbsErrBlock{i},3) - repmat(Y_wide,[1,1,(nb+1)]));
            end
        end
        % Re-collect dummy-coded responses
        Ypred_post{i}(:,class==0) = Ypred{i}(:,dummy==0);
        for j = find(class==1)
            [~, Ypred_post{i}(:,j)] = max(Ypred{i}(:,(sum(Ywidth(1:(j-1)))+1) : sum(Ywidth(1:j))),[],2);
        end
        dims   = ind2subS(ncomps, i);
        Yp = cellget(Ypred_post, dims);
        if sum(class==0)>0
            error  = mat_ins_class(error,'(', dims, ')', sqrt(mean((Y_orig(:,class==0) - Yp(:,class==0)).^2)),class,0);
        end
        if sum(class==1)>0
            error  = mat_ins_class(error,'(', dims, ')', mean(Y_orig(:,class==1) ~= Yp(:,class==1)),class,1);
        end
    end
    if compute == 1 && LDA == 1
        if sum(class==1)>0
            error  = mat_ins_class(error,'(', dims, ')', mean(Y_orig ~= Y_class{i}),class,1);
        end
    end
end
if LDA == 1
    Ypred = Y_class;
else
    Ypred = Ypred_post;
end
%% Insert using vector (ikke i bruk for øyeblikket)
% function X = mat_ins(X,par1,vec,par2,ins)
% expr = ['X' par1];
% for i=1:length(vec)
%     expr = [expr num2str(vec(i)) ','];
% end
% expr = [expr ':' par2];
% expr = [expr '=ins;'];
% eval(expr);


%% Insert using vector
function X = mat_ins_class(X,par1,vec,par2,ins,class,which_class)
expr = ['X' par1];
for i=1:length(vec)
    expr = [expr num2str(vec(i)) ','];
end
expr = [expr 'class==' num2str(which_class) par2];
expr = [expr '=ins;'];
eval(expr);


%% Get cell using vector
function mat = cellget(X, dims)
expr = ['mat = X{' num2str(dims(1))];
for i=2:length(dims)
    expr = [expr ',' num2str(dims(i))];
end
expr = [expr '};'];
eval(expr);


%% Insert matrix in cells
function X = cellinsert(X, dims, cv, a, mat)
expr = ['X{' num2str(dims(1))];
for i=2:length(dims)
    expr = [expr ',' num2str(dims(i))];
end
expr = [expr '}(cv==a,:)'];
expr = [expr ' = mat;'];
eval(expr);


%% Insert matrix in cells
function X = cellinsert3(X, dims, cv, a, mat)
expr = ['X{' num2str(dims(1))];
for i=2:length(dims)
    expr = [expr ',' num2str(dims(i))];
end
expr = [expr '}(cv==a,:,:)'];
expr = [expr ' = mat;'];
eval(expr);


%% Index to subindex, short
function vec = ind2subS(siz,IND)
ndim = length(siz);
vec  = ones(1,20);
[vec(1) vec(2) vec(3) vec(4) vec(5) vec(6) vec(7) vec(8) vec(9) vec(10) ...
    vec(11) vec(12) vec(13) vec(14) vec(15) vec(16) vec(17) vec(18) vec(19) vec(20)] ...
    = ind2sub(siz,IND);
vec = vec(1,1:ndim);


%% Subset of cells
function [Xin, Xout] = cellsub(X,nb,cv,a)
Xin  = X;
Xout = X;
for i = 1:nb
    Xin{i}  = X{i}(cv~=a,:);
    Xout{i} = X{i}(cv==a,:);
end

%% Cross-validation segments
function [cv, s] = cvseg(n,s,type)
% [cv, s] = cvseg(n,s,type)
% Function for making crossvalidation segments
% n - number of samples
% s - number of segments
% type - type of crossvalidation
%		1='Full crossvalidation (leave one out)'
%		2='Random crossvalidation (samles are randomly picked for each segment)'
%		3='Systematic crossvalidation 111..-222..-333.. etc.
%		4='Systematic crossvalidation 123..-123..-123.. etc.
str=ceil(n/s);
if type==1 || s==n %full CV [123...n]
    cv=1:n;
    type=1;
    s = n;
end
if type == 2 %CV-segm randomly chosen
    cv=repmat(1:s,1,str);
    r=randperm(n);
    cv=cv(r);
end
if type==3 %CV-segm = [111 222 333 etc.]
    cv=repmat(1:s,1,str);
    cv=sort(cv(1:n));
end
if type==4 %%CV-segm = [123 123 123 etc.]
    cv=repmat(1:s,1,str);
    cv=cv(1:n);
end

%% Match arguments and defaults
function varargout = match_arguments(names,default_values,varargin)
varargout = default_values;
for i=1:2:length(varargin)
    pos = find(strcmp(varargin{i},names));
    if isempty(pos)
        error('Supplied argument not matched in names')
    else
        varargout{pos} = varargin{i+1};
    end
end

%% Make dummy response
function dum = dummyfy(X)

n = size(X,1);
p = max(X);
dum = zeros(n,p);
for i=1:n
    dum(i,X(i))=1;
end