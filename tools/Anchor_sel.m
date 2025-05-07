function [aind] = Anchor_sel(W,nCluster)

    n = size(W,2);
    rho = zeros(1,n);
    
    for i=1 : n-1
        for j=i+1 : n
           if(W(i,j) > 0.5)
                rho(i) = rho(i) + W(i,j);
                rho(j) = rho(j) + W(i,j);
           end
        end
    end

    dist = ones(n)./(W+eps);
    maxd = max(max(dist));

    [~,ordrho]=sort(rho,'descend'); 
    delta(ordrho(1))= -1; 
    nneigh(ordrho(1))=0;

    for ii=2:n
       delta(ordrho(ii))=maxd;
       for jj=1:ii-1
         if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
            delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
            nneigh(ordrho(ii))=ordrho(jj); 
         end 
       end
    end

    delta(ordrho(1))=max(delta(:));
    gamma = zeros(1,n);
    for i=1:n
      gamma(i) = rho(i)*delta(i);
    end
    [~,gamma_sortedInd] = sort(gamma,'descend');
    aind = gamma_sortedInd(1:nCluster);    
end