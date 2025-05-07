clear; close all; clc;
addpath(genpath('./'));
datadir='./datasets/';
load('Wiki_fea.mat');

nv = length(X);            % The number of views
n = length(Y);             % The number of samples
nc = length(unique(Y));    % The number of clusters
rho = 1; % 0.4             % The training ratio
pa = 0.5;                  % The proportion of anchors

%% Parameter settings
lambda = 10;

%% Normalization
for v = 1:nv
    X{v} = X{v}';
end

%% Generate the in-sample and out-of-sample data
IZ = []; IX=cell(1,nv);
if rho<1
    [Idi,Ido,NY,ni] = GIOS(Y,nc,rho);
    m = [round(pa*ni)];
    OZ = []; OX = cell(1,nv);
    for v = 1:nv
        IX{v} = X{v}(:,Idi);
        OX{v} = X{v}(:,Ido);
        IZ = [IZ;IX{v}];
        OZ = [OZ;OX{v}];
    end
else
    m = round(pa*n);
    for v = 1:nv
        IX{v} = X{v};
        IZ = [IZ;IX{v}];
    end
    NY = Y;
end

%% Construct the similarity graph
options.k = 5;
options.WeightMode = 'HeatKernel';
SG = constructW(IZ', options);
[inds] = Anchor_sel(SG,m);           % The indices of the selected concatenated anchors
W = SG(inds,inds);
D = diag(sum(W,2));
L = D-W;                             % Graph Laplacian
CA = IZ(:,inds);                     % Concatenated anchors
IG = exp(-EuDist2(CA',IZ',1)/5);     % Anchor graph of the in-sample data
if rho<1
    OG = exp(-EuDist2(CA',OZ',1)/5); % Anchor graph of the out-of-sample data
    G = [IG,OG];
else
    G = IG;
end
G = G./repmat(sqrt(sum(G.^2,1))+eps,size(G,1),1);
A = cell(1,nv);
for v = 1:nv
    A{v} = IX{v}(:,inds);            % Anchors of each view
end

%% Run APS-MVC
tic;
[GtSt,prelabel] = APS_MVC(A,G,L,nv,nc,lambda);
time = toc;
result = measurement(prelabel,NY);
fprintf('\n Results: ACC: %.4f, NMI: %.4f, F: %.4f, ARI: %.4f', result(1), result(2), result(3), result(4));

