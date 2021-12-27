function SO_PLS_plot(SO, type, block, ncomps, varargin)
%% Generic plotting tool for SO-PLS
% EXAMPLE:
% SO_PLS_plot(SO, 'T', 1, 5)
% 
% ARGUMENTS:
% SO     - Object fitted by SO-PLS
% type   - What to plot, e.g. W, T, P, V
% block  - Block number to be plotted
% ncomps - Model of choice (component combination)
% 
% EXTRA:
% dimensions - Which dimensions to plot (default = 1:2)
% spectra    - Plot as spectra (default = 0)
% subset     - Plot a subsets of the data (cell of indexes)
% symbols    - Which plot symbols to use
% colors     - User defined colors (matrix)
% legend     - *Not implemented
% xvalues    - Values of x dimension of spectra, e.g. wavelengths
% labels     - Automatic axis labeling and title (default = 1)

% Extra arguments and defaults
names = {'dimensions' 'spectra' 'subset' 'symbols' 'colors' 'legend' 'xvalues' 'labels' 'block' 'zeronan' 'revx'};
dflts = {        1:2         0       []        'o'      []        1        []        1       []         0  false};
[dimensions,spectra,subset,symbols,colors,legend,xvalues,labels,bname,zeronan,revx] = match_arguments(names,dflts,varargin{:});

if max(dimensions) > max(ncomps)
    dimensions = 1;
    if spectra == 0
        spectra = 1;
    end
end
dimensions = dimensions + 1; % Compensate for 0 component models
if length(ncomps) == 1
    error('ncomps must specify number of components for all blocks')
end
X = SO_PLS_extract(SO, type, block, ncomps+1);
if zeronan == 1
    X(X==0) = NaN;
end
if isempty(subset)
    has_subset = 0;
else
    has_subset = 1;
end
if isempty(colors) || size(colors,2) == 1
    colors =  colormap('Lines');
end
if isempty(xvalues) && spectra == 1% && has_subset == 0
    xvalues = 1:size(X,1);
end
linestyles = {'-','--','.-',':','-.'};

% Ordinary plotting (no subsets)
if has_subset == 0
    if spectra == 1 % Plot results as spectra
        plot(xvalues,X(:,dimensions(1)),'-','Color',colors(1,:))
        if length(dimensions)>1
            hold on
            for i=2:length(dimensions)
                plot(xvalues,X(:,dimensions(i)),'-','Color',colors(dimensions(i),:))
            end
        end
        
    else % Plot as scatter
        plot(X(:,dimensions(1)),X(:,dimensions(2)),symbols(1),'Color',colors(1,:))
        if ~isempty(colors) && size(colors,2) == 1
            hold on
            for i=2:max(colors)
                plot(X(colors==i,dimensions(1)),X(colors==i,dimensions(2)), symbols(1),'Color',Colors(i,:));
            end
        end
    end
    
    % Plot subsets
else
    hold on
    for j=1:length(subset)
        if spectra == 1 % Plot results as spectra
            plot(xvalues(subset{j}),X(subset{j},dimensions(1)),linestyles{j},'Color',Colors(1,:))
            if length(dimensions)>1
                for i=2:length(dimensions)
                    plot(xvalues(subset{j}),X(subset{j},dimensions(i)),linestyles{j},'Color',Colors(dimensions(i),:))
                end
            end
            
        else % Plot as scatter
            plot(X(subset{j},dimensions(1)),X(subset{j},dimensions(2)),symbols(min(length(symbols),j)),'Color',Colors(j,:))
        end
    end
end

if labels == 1
    if spectra ~= 1
        xlabel(['Component ' num2str(dimensions(1)-1)])
        ylabel(['Component ' num2str(dimensions(2)-1)])
    else
        xlabel('Variable')
        ylabel('Intensity')
    end
  
    % X scores
    if (strcmp(type, 'T') || strcmp(type, 'scores'))
        title(['Score plot - block ' num2str(block)])
    end
    % X loadings
    if strcmp(type, 'P') || strcmp(type, 'PX') || strcmp(type, 'loadings')
        if isempty(bname)
            title(['Loading plot - block ' num2str(block)])
        else
            title(['Loading plot - ' bname])
        end
    end
    % X loading weights
    if strcmp(type, 'W') || strcmp(type, 'loadingweights')
        if isempty(bname)
            title(['Loading weights plot - block ' num2str(block)])
        else
            title(['Loading weights plot - ' bname])
        end
    end
    % X projections
    if strcmp(type, 'V') || strcmp(type, 'projections')
        title(['X projections plot - block ' num2str(block)])
    end
    % Y loadings
    if strcmp(type, 'Q') || strcmp(type, 'yloadings')
        title(['Y loading - block ' num2str(block)])
    end
    % Beta coefficients
    if strcmp(type, 'B') || strcmp(type, 'coefficients')
        title(['Regression coefficients - block ' num2str(block)])
    end
end
if revx
    set(gca,'XDir','rev')
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
