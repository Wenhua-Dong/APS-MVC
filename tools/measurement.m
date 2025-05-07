function [result] = measurement(label,gnd)

[A,NMI,avgent] = compute_nmi(gnd,label);
[F,P,R] = compute_f(gnd,label);
ARI = RandIndex(gnd,label);
label = bestMap(gnd,label);
ACC = length(find(gnd == label))/length(gnd);
result = [ACC,NMI,F,ARI];
end