function SO = SO_PLS_from_rec(SO_rec, ncomps)

% Copy and initialize
SO = SO_rec;
SO.W = cell(1,SO.nb);
SO.P = cell(1,SO.nb);
SO.Q = cell(1,SO.nb);
SO.T = cell(1,SO.nb);
SO.V = cell(1,SO.nb);
SO.E = cell(1,SO.nb);
SO.pow = cell(1,SO.nb);

% Extract correct model
for i = 1:SO.nb
    ncomp = ncomps;
    ncomp(i+1:end) = zeros(1,length(ncomp(i+1:end)));
    for j = 0:ncomps(i)
        ncomp(i) = j;
        SO.W{i} = [SO.W{i} vec_extr(SO_rec.W,'{', ncomp+1, '}', i)];
        SO.P{i} = [SO.P{i} vec_extr(SO_rec.P,'{', ncomp+1, '}', i)];
        SO.Q{i} = [SO.Q{i} vec_extr(SO_rec.Q,'{', ncomp+1, '}', i)];
        SO.T{i} = [SO.T{i} vec_extr(SO_rec.T,'{', ncomp+1, '}', i)];
        SO.V{i} = [SO.V{i} vec_extr(SO_rec.V,'{', ncomp+1, '}', i)];
        SO.E{i} = [SO.E{i} vec_extr(SO_rec.E,'{', ncomp+1, '}', i)];
        SO.pow{i} = [SO.pow{i} vec_extr(SO_rec.pow,'{', ncomp+1, '}', i)];
    end
end

% Set number of components equal to chosen model
SO.ncomps = ncomps;


%% Extract using vector
function ret = vec_extr(X,par1,vec,par2,nc)
expr = ['X' par1];
for i=1:length(vec)
    expr = [expr num2str(vec(i)) ','];
end
expr = [expr num2str(nc) par2];
% expr = [expr '=[' expr ' ins];'];
ret = eval(expr);

%% Insert matrix in cells
function X = cellinsert(X, dims, cv, a, mat)
expr = ['X{' num2str(dims(1))];
for i=2:length(dims)
    expr = [expr ',' num2str(dims(i))];
end
expr = [expr '}(cv==a,:)'];
expr = [expr ' = mat;'];
eval(expr);