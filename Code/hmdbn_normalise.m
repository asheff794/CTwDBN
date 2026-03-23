function [M, z] = hmdbn_normalise(A, dim)
% NORMALISE Make the entries of a (multidimensional) array sum to 1
% [M, z] = hmdbn_normalise(A)
% z is the normalizing constant
%
% [M, z] = normalise(A, dim)
% If dim is specified, we normalise the specified dimension only,
% otherwise we normalise the whole array.

if nargin < 2
  z = sum(A(:));
  % Set any zeros to one before dividing
  % This is valid, since c=0 => all i. A(i)=0 => the answer should be 0/1=0
  s = z + (z==0);
  M = A / s;
elseif dim==1 % normalize each column
  z = sum(A,dim);
  s = z + (z==0);
  %M = A ./ (d'*ones(1,size(A,1)))';
  for i=1:size(A,1)
    M(i,:) = A(i,:) ./ s;
  end
else
  % Keith Battocchi - v. slow because of repmat
  z=sum(A,dim);
  s = z + (z==0);
  L=size(A,dim);
  d=length(size(A));
  v=ones(d,1);
  v(dim)=L;
  %c=repmat(s,v);
  c=repmat(s,v');
  M=A./c;
end


