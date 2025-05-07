function [GtSt,prelabel] = APS_MVC(A,G,L,nv,nc,lambda)

% ~~~~~~~~~~~~~~~~~~~~~~~~ Objective function ~~~~~~~~~~~~~~~~~~~~
% min sum_{v=1}^{nv} (||alpha^{v}A^{v}-C^{v}S||_F^{2}+lambda*Tr(SLS^{T}),
% s.t. C^{v}^{T}C^{v} = I, S>=0, S^{T}1 = 1,
% A^{v}:     The v-th anchor matrix;
% C^{v}:     The v-th clustering centroid matrix;
% S:         Consistent soft partition of anchors;
% alpha^{v}: Control parameter;
% lambda:    The trade-off parameter.
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

m = size(A{1},2);
maxIter = 20 ;     % The number of iterations
C = cell(1,nv);
%% Initialization
S = zeros(nc,m);

%% Run APS-MVC
for iter = 1:maxIter
    %% Solve C^{v}
    parfor v = 1:nv
        tmp1 = S*A{v}';
        [Uc,~,Vc] = svd(tmp1,'econ');
        Ct = Uc*Vc';
        C{v} = Ct';
    end

    %% Solve S
    tmp2 = nv*eye(m)+lambda*L;
    tmp3 = 0;
    for v = 1:nv
        tmp3 = tmp3+C{v}'*A{v}/nv;
    end
    R = chol(tmp2, 'lower')+eps;
    invtmp2 = R\(R'\eye(m));
    tS = tmp3*invtmp2;
    parfor i = 1:m
        si = tS(:,i);
        S(:,i) = EProjSimplex_new(si');
    end

    %% Calculate the value of the objective function
    loss = 0;
    for v = 1:nv
        E = A{v}/nv - C{v}*S;
        loss = loss+sum(sum(E.*E, 1));
    end
    loss = loss + lambda*(sum(diag(S*L*S')));
    obj(iter) = loss;

    if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-7)
        break;
    end

end
GtSt = G'*S';
GtSt = GtSt./repmat(sum(GtSt,1)+eps,size(GtSt,1),1);
GtSt = GtSt./repmat(sum(GtSt,2)+eps,1,nc);
[~, prelabel] = max(GtSt,[],2);
end

