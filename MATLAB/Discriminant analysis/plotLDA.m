function plotLDA(w, Xtrain, Ytrain, Xtest, Ytest)

lines = 1;

% Preparation
nGr = max(Ytrain);
Y = dummy(Ytrain);
[n,m] = size(Y);
S = Y/(Y'*Y)*Y';

% Score vectors
T = Xtrain*w;
if nargin > 3
    Tt = [Xtrain(1,:);Xtest]*w;
end

if size(w,2) == 2
    Xinn = T;
    if nargin > 3
        Xinnt = Tt(2:end,:);
    end
else
    B = 1/n*T'*S*T;                           % Between groups covariance
    To = 1/(n-1)*T'*(eye(n)-1/n*ones(n,n))*T; % Total centered covariance
    %To = 1/(n-1)*T'*T;                       % Total uncentered covariance
    Wa = To - B;                              % Within groups covariance
    Wa(abs(Wa)<eps) = 0;
    WB = inv(Wa)*B;
    [a,b] = eigs(WB,max(1,2));

    XX = T*a(:,1); XX = XX*sign(XX(1,1));   % Discriminant vector 1 (training)
    XY = T*a(:,2); XY = XY*sign(XY(1,1));   % Discriminant vector 2 (training)
    Xinn = [XX, XY];
    if nargin > 3
        XXt = Tt*a(:,1); XXt = XXt*sign(XXt(1,1));   % Discriminant vector 1 (test)
        XYt = Tt*a(:,2); XYt = XYt*sign(XYt(1,1));   % Discriminant vector 2 (test)
        XXt(1,:) = []; XYt(1,:) = [];
        Xinnt = [XXt, XYt];
    end
end

%% Plot objects
hold on
symbs = {'x', 's', '*', '^', '+', 'd', '^', 'v'};
for j=1:n
    symb = symbs{Ytrain(j)}; % symbs{floor((i-1)/6)+1}
    plot(Xinn(j,1),Xinn(j,2),symb,'Color',niceC(mod(Ytrain(j)-1,6)+1))
end

% for i=1:nGr
%     symb = symbs{i}; % symbs{floor((i-1)/6)+1}
%     plot(Xinn(Ytrain==i,1),Xinn(Ytrain==i,2),symb,'Color',niceC(mod(i-1,6)+1))
% end

if nargin == 5
    for i=1:nGr
        plot(Xinnt(Ytest==i,1),Xinnt(Ytest==i,2),'.','Color',niceC(i))
    end
elseif nargin == 4
    plot(Xinnt(:,1),Xinnt(:,2),'.k')
else
    Xinnt = Xinn;
end

%% Prepare LDA
% N = sum(Y)';       % Group sizes
% Pinn = ones(1,m)./m;
Pinn = sum(Y)./sum(sum(Y));

[n,p] = size(Xinn); % Dimension of training data

% Preparation
Xs = (Y'*Y)\Y'*Xinn;   % Mean for each group
Xc = Xinn-Xs(Ytrain,:);           % Centering
Si = inv(Xc'*Xc./(n-m));  % Inverted common covariance matrix

% Minimum and maximum values of observations
minX1 = min([Xinn(:,1);Xinnt(:,1)]);
maxX1 = max([Xinn(:,1);Xinnt(:,1)]);
miX1 = minX1 - (maxX1-minX1)*0.05;
maX1 = maxX1 + (maxX1-minX1)*0.05;
minX2 = min([Xinn(:,2);Xinnt(:,2)]);
maxX2 = max([Xinn(:,2);Xinnt(:,2)]);
miX2 = minX2 - (maxX2-minX2)*0.05;
maX2 = maxX2 + (maxX2-minX2)*0.05;

% Line functions
Xut1 = inline('(M-Xinn*L2)/L1','Xinn','M','L1','L2');
Xut2 = inline('(M-Xinn*L1)/L2','Xinn','M','L1','L2');
Xkryss1 = inline('(Mij/Lij1-Mkl/Lkl1)/(Lij2/Lij1-Lkl2/Lkl1)',...
    'Mkl','Mij','Lkl1','Lkl2','Lij1','Lij2');
Xkryss2 = inline('(Mij/Lij2-Mkl/Lkl2)/(Lij1/Lij2-Lkl1/Lkl2)',...
    'Mkl','Mij','Lkl1','Lkl2','Lij1','Lij2');

% Lines
if lines == 1
    M = zeros(m,m);
    L1 = zeros(m,m);
    L2 = zeros(m,m);
    for i=1:m-1
        for j=i+1:m
            M(i,j) = 0.5*(Xs(i,:)-Xs(j,:))*Si*(Xs(i,:)+Xs(j,:))'-log(Pinn(i)/Pinn(j));
            L1(i,j) = Si(1,1)*(Xs(i,1)-Xs(j,1)) + Si(1,2)*(Xs(i,2)-Xs(j,2));
            L2(i,j) = Si(2,1)*(Xs(i,1)-Xs(j,1)) + Si(2,2)*(Xs(i,2)-Xs(j,2));
        end
    end

    % Crosses between lines ij and ik
    K1 = ones(6,6,6,6)*NaN;
    K2 = ones(6,6,6,6)*NaN;
    for i=1:m
        for j=i+1:m
            for k=1:m
                for l=k+1:m
                    if ~((i==k && j==l) || (i==l && j==k))
                        K1(i,j,k,l) = Xkryss1(M(k,l),M(i,j),L1(k,l),L2(k,l),L1(i,j),L2(i,j));
                        K2(i,j,k,l) = Xkryss2(M(k,l),M(i,j),L1(k,l),L2(k,l),L1(i,j),L2(i,j));
                    end
                end
            end
        end
    end

    % Drawing line segments based on discriminant values and crosses
    d = zeros(m,1);
    for i=1:m-1
        for j=i+1:m
            if(abs(L1(i,j)/L2(i,j)) > 1) % Steep lines
                % Crosses of line ij
                R = squeeze(K1(i,j,:));
                RR = [sort(R(~isnan(R(:))));maX2];
                a = miX2;
                for t=1:length(RR)
                    % Checks if midpoint to next cross has (i,j) as largest
                    % discriminant values
                    y = (RR(t)+a)/2;
                    x = Xut1(y,M(i,j),L1(i,j),L2(i,j));
                    for o=1:m
                        d(o,1) = Xs(o,:)*Si*[x,y]' - 0.5*Xs(o,:)*Si*Xs(o,:)' + log(Pinn(o));
                    end
                    [sor,rek] = sort(d,1,'descend');
                    if sum(rek(1:2)==[i;j]) == 2 || sum(rek(1:2)==[j;i]) == 2
                        line([Xut1(a,M(i,j),L1(i,j),L2(i,j)),Xut1(RR(t),M(i,j),L1(i,j),L2(i,j))],[a,RR(t)],'Color',[0,0,0])
                    end
                    a = RR(t);
                end
            else % Non-steep lines
                R = squeeze(K2(i,j,:));
                RR = [sort(R(~isnan(R(:))));maX1];
                a = miX1;
                for t=1:length(RR)
                    % Checks if midpoint to next cross has (i,j) as largest
                    % discriminant values
                    x = (RR(t)+a)/2;
                    y = Xut2(x,M(i,j),L1(i,j),L2(i,j));

                    for o=1:m
                        d(o,1) = Xs(o,:)*Si*[x,y]' - 0.5*Xs(o,:)*Si*Xs(o,:)' + log(Pinn(o));
                    end
                    [sor,rek] = sort(d,1,'descend');
                    if sum(rek(1:2)==[i;j]) == 2 || sum(rek(1:2)==[j;i]) == 2
                        line([a,RR(t)],[Xut2(a,M(i,j),L1(i,j),L2(i,j)),Xut2(RR(t),M(i,j),L1(i,j),L2(i,j))],'Color',[0,0,0])
                    end

                    a = RR(t);
                end
            end
        end
    end
end

ylim([miX2,maX2])
xlim([miX1,maX1])
% set(gca, 'ytick',[])
% set(gca, 'xtick',[])


function [c] = niceC(i)
C = [200   0   0
    200 200   0
    0 200   0
    0 200 200
    0   0 200
    200   0 200];
% C = [255    10    10
%      0   170     0
%     20    20   200
%    200    20   130
%    140    50    20
%    200   160    20
%    150   150   150];
c = C(i,:)./255;