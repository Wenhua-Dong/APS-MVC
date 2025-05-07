function [Idi,Ido,NY,nin] = GIOS(Y,k,rho)

Idi =[]; Ido =[];  
rng(0)
for i = 1:k    
    ind = find(Y==i);
    ni = round(rho*length(ind)); 
    index = randperm(length(ind),ni); 
    Idi = [Idi;ind(index)];    
    ind(index) = [];
    Ido =[Ido;ind]; 
end
nin = length(Idi);
NY = Y([Idi;Ido]);
