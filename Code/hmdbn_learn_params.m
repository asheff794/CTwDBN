function CPD= hmdbn_learn_params(fam,ns, data,gammaG)
%function CPD = hmdbn_learn_params(fam,ns, data,gammaG)
% LEARN_PARAMS Compute the ML/MAP estimate of the params of a tabular CPD given complete data
% 
% fam = [ps j]
% gammaG is the p(O,q=Gij|HMDBN) for node j
% Learn tabular CPD for parameter for node j
% 

local_data = data(fam, :); 
if iscell(local_data)
  local_data = cell2num(local_data);
end
sz=ns(fam);

counts= hmdbn_compute_counts(local_data, sz,gammaG);

CPD= mk_stochastic(counts); 
 
end
