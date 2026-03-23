function count = hmdbn_compute_counts(local_data, sz, gammaG)
% COMPUTE_COUNTS Count the number of times each combination of discrete assignments occurs
% count = hmdbn_compute_counts(local_data, sz, gammaG)
%
% data(i,) is the value of variable i 
% sz(i) : values for variable i are assumed to be in [1:sz(i)]


assert(length(sz) == size(local_data, 1));
P = prod(sz);
indices = subv2ind(sz, local_data'); % each row of data' is a case 
%count = histc(indices, 1:P);
count=zeros(1,P);
for(i =1:P)
    count(i)=sum(   gammaG(find(indices==i))+1   );
end

%count = hist(indices, 1:P);
count = myreshape(count, sz);

