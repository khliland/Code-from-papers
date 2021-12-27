function [corrected, parameters, par_names, model] = emsc(X, varargin)
%% EMSC for hyperspectral images 
% [corrected, parameters, par_names, model] = hyspec_emsc(X, varargin)
% 
% --= Compulsory argument =--
% X:
%     a matrix of size samples x spectra
% 
% --= Optional arguments =--
% block_size (default = [100,100]):
%      size of blocks in blockproc
% reference (default = mean spectrum):
%      reference spectrum
% terms (default = 4): 
%     (1)Baseline, (2)Reference,    (3)Linear,      (4)Quadratic,
%     (5)Cubic,    (6)Fourth order, (7)Fifth order, (8)Sixth order
% constituent (default = none):
%     constituent spectra
% interferent (default = none):
%     interferent spectra
% abscissas (default = 1:p):
%     wavelengths, wavenumbers, shifts, ... (numeric)
% weights (default = none)
%     weighting of wavelengths/variables
% model (default = none)
%     previously calculated model, e.g. for used on new data
% 
% --= Output =--
% corrected:
%     corrected spectra from EMSC
% parameters:
%     estimated EMSC parameters per spectrum
% par_names:
%     names of parameters in EMSC model
% model:
%     model spectra used in the EMSC

% Extra arguments and defaults
names = {'block_size' 'reference' 'terms' 'constituent' 'interferent' 'abscissas' 'weights' 'model'};
dflts = {   [100,100]          []       4            []            []          []        []      []};
[block, reference, terms, constituent, interferent, abscissas, weights, model] = match_arguments(names,dflts,varargin{:});

global parallel
if isempty(parallel)
    parallel = false;
end

[n,p] = size(X);
I = reshape(X, [1,n,p]);
i = 1;

% Reference calculation
if isempty(reference) && terms > 1
    reference = hyspec_mean(I);
    if i > 1
        reference = mean(reference);
    end
end

% Artificial abscissas
if isempty(abscissas)
    if isstruct(I)
        abscissas = I(1).v;
    else
        abscissas = 1:p;
    end
end

if isempty(model)
    % Prepare parameter matrix
    [model] = EMSC_param(terms, reference, ...
        constituent, interferent, abscissas, p);
end
model.weights = weights;
model.terms   = terms;
par_names     = model.par_names;

% Block function
fun_emsc = @(block_struct) EMSC(block_struct.data, model);

h = waitbar(0,'(E)MSC', 'Name', '(E)MSC');
% Correction
[r,c,~,~] = size(I);
corrected  = zeros(r,c,p,i);
parameters = zeros(r,c,size(par_names,1),i);

for j=1:i
    waitbar(j/i,h,['Correcting image: ' num2str(j) '/' num2str(i)]);
    
    % Apply block function
    M = blockproc(I(:,:,:,j),block,fun_emsc, 'UseParallel', parallel);
    
    % Determine size and collect block calculations
    corrected(:,:,:,j)  = M(:,:,1:p);
    parameters(:,:,:,j) = M(:,:,(p+1):end);
end
corrected  = reshape(corrected,[n,p]);
parameters = reshape(parameters,[n,size(parameters,3)]);
close(h)


%% EMSC calculations
function out = EMSC(I, MModel)

% Initializsation
model   = MModel.model;
sizes   = MModel.sizes;
weights = MModel.weights;
terms   = MModel.terms;
[r,c,p] = size(I);
M = size(model,2);
I = reshape(I,r*c,p);
Corrected  = zeros(r*c,p);
Parameters = zeros(r*c,M);

% Apply weights (if included)
if ~isempty(weights)
    Iw     = bsxfun(@times,I,weights);
    modelw = bsxfun(@times,model,weights);
else
    Iw     = I;
    modelw = model;
end

% Handling of 0 objects
PP = all(Iw==0,2);

if any(PP)
    Parameters0  = (modelw\Iw(~PP,:)')';
    
    k = setdiff(1:terms,2);
    Corrected0 = I(~PP,:)-Parameters0(:,k)*model(:,k)';
    if (sizes(3)>0)
        k=(terms+sizes(2)+1):M;   % bad spectra
        Corrected0 = Corrected0-Parameters0(:,k)*model(:,k)';
    end
    if(sizes(1)>1)
        Corrected0=bsxfun(@times,Corrected0,1./Parameters0(:,2)); % correct multipl. eff.
    end
    
    Corrected(PP,:)   = I(PP,:);
    Corrected(~PP,:)  = Corrected0;
    Parameters(~PP,:) = Parameters0;
    Parameters(PP,:)  = 0;

else
    Parameters0  = (modelw\Iw')';
    
    k = setdiff(1:terms,2);
    Corrected0 = I-Parameters0(:,k)*model(:,k)';
    if (sizes(3)>0)
        k=(terms+sizes(2)+1):M;   % bad spectra
        Corrected0 = Corrected0-Parameters0(:,k)*model(:,k)';
    end
    if(sizes(1)>1)
        Corrected0=bsxfun(@times,Corrected0,1./Parameters0(:,2)); % correct multipl. eff.
    end
    
    Corrected  = Corrected0;
    Parameters = Parameters0;
end

out = reshape([Corrected Parameters],[r,c,p+sum(sizes)]);


%% EMSC parameter prep
function [MModel] = EMSC_param(terms, reference, constituent, interferent, abscissas, p)

WaveNum = abscissas;
Start   = WaveNum(1);
End     = WaveNum(p);
WaveNumT    = WaveNum';

C  = 0.5*(Start+End);
M0 = 2.0/(Start-End);
M  = 4.0/((Start-End)*(Start-End));

model = ones(terms,p); % Baseline
if terms > 1
    model(2,:) = reference;
    if terms > 2
        model(3,:) = M0*(Start-WaveNumT)-1;
        if terms > 3
            model(4,:) = M*(WaveNumT-C).^2;
            if terms > 4
                model(5,:) = M*(1/(Start-End))*(WaveNumT-C).^3;
                if terms > 5
                    model(6,:) = M*M*(WaveNumT-C).^4;
                    if terms > 6
                        model(7,:) = M*M*M0*(WaveNumT-C).^5;
                        if terms > 7
                            model(8,:) = M*M*M*(WaveNumT-C).^6;
                        end
                    end
                end
            end
        end
    end
end

%  Add constituent and/or interferent spectra
model = [model; constituent; interferent]';

% Names for model components
par_names = ['Baseline        '; ...
        'Reference       '; ...
        'Linear          '; ...
        'Quadratic       '; ...
        'Cubic           '; ...
        'Fourth order    '; ...
        'Fifth order     '; ...
        'Sixth order     '];
par_names = par_names(1:terms,:);
n_con = size(constituent,1);
n_int = size(interferent,1);
if n_con > 0
    par_names = [par_names; [repmat('Constituent ', n_con, 1) num2str((1:n_con)') repmat(' ', n_con, 3-(n_con>9))]];
end
if n_int > 0
    par_names = [par_names; [repmat('Interferent ', n_int, 1) num2str((1:n_int)') repmat(' ', n_int, 3-(n_int>9))]];
end

MModel.model       = model;
MModel.sizes       = [terms, n_con, n_int]; 
MModel.par_names   = par_names;


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

function M = hyspec_mean(I, varargin)
% M = hyspec_mean(I, varargin)
%
% Calculate the mean spectrum across all pixels (per image)
% I = array of dimensions (pixels x pixels x spectra (x images))
%     or hyspec_object
% 
% --= Optional arguments =--
% block_size (default = [100,100]):
%      size of blocks in blockproc

% Extra arguments and defaults
names = {'block_size'};
dflts = {   [100,100]};
[block] = match_arguments(names,dflts,varargin{:});

global parallel
if isempty(parallel)
    parallel = false;
end

% Initialize
if isstruct(I)
    i = length(I);
    p = size(I(1).d,3);
else
    i = size(I,4);
    p = size(I,3);
end
M = zeros(i,p);

% Block function
% fun_sum = @(block_struct) squeeze(sum(sum(block_struct.data),2));
fun_sum = @(block_struct) squeeze(nanmean(nanmean(block_struct.data),2));

for j=1:i
    % Apply block function
    if isstruct(I)
        I2 = blockproc(I(j).d,block,fun_sum,'UseParallel',parallel);
    else
        I2 = blockproc(I(:,:,:,j),block,fun_sum,'UseParallel',parallel);
    end
    
    % Determine size and collect block calculations
    reps   = size(I2,1)/p;
%     M(j,:) = sum(reshape(I2,p,reps*size(I2,2)),2)./(r*c);
    M(j,:) = nanmean(reshape(I2,p,reps*size(I2,2)),2);
end