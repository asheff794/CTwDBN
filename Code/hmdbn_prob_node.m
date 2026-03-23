function p = hmdbn_prob_node(CPD, self_ev, pev)
% PROB_NODE Compute P(y|pa(y), theta) (tabular)
% p = hmdbn_prob_node(CPD, self_ev, pev) 
% This function compute the probability for only one node,
% the probability is a vector, where each value corresponds to one time step
% self_ev is the evidence on this node
% pev(i,) is the evidence on the i'th parent
% 

ncases = size(pev, 2);

%assert(~any(isemptycell(pev))); % slow
%assert(~any(isemptycell(self_ev))); % slow

%CPT = CPD_to_CPT(CPD);  
sz =size(CPD);
if(sz(2)==1)
   sz=[2]; 
end
nparents = length(sz)-1;
%assert(nparents == size(pev, 1));


  x = (pev)'; % each row is a case
  y = (self_ev)';
  ind = subv2ind(sz, [x y]);
  p = CPD(ind);
  %p=p.^20;
end     
