% Måge-plott
function [] = maage(RMSEP, varargin)

% Extra arguments and defaults
names = {'trace0' 'tracelegend' 'ymax'};
dflts = {      0             0      0 };
[trace0,tracelegend,ymax] = match_arguments(names, dflts, varargin{:});

if ymax == 0
    ymax = max(RMSEP(:))*1.05;
end

n = size(RMSEP);
nd = ndims(RMSEP);

hold on
if trace0 ~= 0 % Plot trace of pure block predictions
    C = colormap(lines(nd));
    for i=1:nd
        tmp = permute(RMSEP,[i setdiff(1:nd,i)]);
        plot(0:size(tmp,1)-1,tmp(:,1),'Color',C(i,:))
    end
    if length(tracelegend)>1
        legend(tracelegend, 'AutoUpdate','off')
    end
end

xmax = 0;
for i=1:numel(RMSEP)
    if ~isnan(RMSEP(i))
        ind = ind2subS(n,i);
        str = num2str(ind(1)-1);
        for j=2:nd
            str = [str '.' num2str(ind(j)-1)];
        end
        x = sum(ind)-nd;
        plot(x,RMSEP(i),'.k')
        text(x + 0.2,RMSEP(i),str,'FontSize',6, 'Clipping', 'on')
        if x > xmax
            xmax = x;
        end
    end
% Plot RMSEP(i) med x-verdi lik sum av komponenter/dimensjoner for i
% og etikett sammensatt av dimensjoner
end
% xmax = max(sum(~isnan(RMSEP))+(1:size(RMSEP,2))-2);
xlim([0 xmax*1.1])
% xlim([0 (sum(n)-nd)*1.1])
ylim([0 ymax])

%% Index to subindex, short
function vec = ind2subS(siz,IND)
ndim = length(siz);
vec  = ones(1,20);
[vec(1) vec(2) vec(3) vec(4) vec(5) vec(6) vec(7) vec(8) vec(9) vec(10) ...
    vec(11) vec(12) vec(13) vec(14) vec(15) vec(16) vec(17) vec(18) vec(19) vec(20)] ...
    = ind2sub(siz,IND);
vec = vec(1,1:ndim);

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
