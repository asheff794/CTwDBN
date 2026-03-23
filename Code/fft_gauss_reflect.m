function Y = fft_gauss_reflect(X, sigma, dim)
%FFT_GAUSS_REFLECT Zero-phase Gaussian smoothing via FFT with reflective padding.
%   Y = FFT_GAUSS_REFLECT(X, SIGMA) smooths X along the first dimension
%   using a Gaussian kernel with standard deviation SIGMA (in samples),
%   implemented in the frequency domain. Reflective padding is used to
%   suppress wrap-around artifacts, and the original length is preserved.
%
%   Y = FFT_GAUSS_REFLECT(X, SIGMA, DIM) smooths along dimension DIM.
%
%   This mirrors the behavior of a zero-phase (non-causal) Gaussian
%   filter like filtfilt with a sufficiently long FIR, but is typically
%   much faster for large signals because it uses FFT-based convolution.
%
%   Inputs:
%     - X: numeric array (time along DIM)
%     - sigma: positive real scalar, std-dev in samples
%     - dim: (optional) dimension to filter (default 1)
%
%   Output:
%     - Y: smoothed array, same size as X
%
%   Notes:
%     - Uses reflective padding by concatenating X with flip(X, DIM).
%     - Applies analytic Gaussian frequency response H(k) = exp(-0.5*(2*pi*sigma*f_k).^2),
%       where f_k are the discrete frequencies for length L (the padded length).
%     - If X is real, the output is forced real via ifft(..., 'symmetric').
%
%   Alec Sheffield, 2025

arguments
    X {mustBeNumeric}
    sigma (1,1) double {mustBeNonnegative}
    dim (1,1) double {mustBeInteger, mustBePositive} = 1
end

if sigma == 0 || all(size(X, dim) <= 1)
    Y = X; % nothing to do
    return;
end

% Ensure double for numerical stability (avoid overflow/underflow)
origClass = class(X);
if ~isa(X, 'double')
    X = double(X);
end

% Reflect-pad along DIM
n = size(X, dim);
Xpad = cat(dim, X, flip(X, dim));
L = size(Xpad, dim);

% Build Gaussian frequency response for length L
% Positive frequency bins (including DC and Nyquist when even): 0:floor(L/2)
Npos = floor(L/2) + 1;
freqs = (0:(Npos-1)) / L; % cycles per sample
Hpos = exp(-0.5 * (2*pi*sigma*freqs).^2);

% Mirror to full spectrum length L
if mod(L, 2) == 0
    % even length: [0 .. L/2] then mirror [L/2-1 .. 1]
    Hfull = [Hpos, Hpos(end-1:-1:2)];
else
    % odd length: [0 .. (L-1)/2] then mirror [(L-1)/2 .. 1]
    Hfull = [Hpos, Hpos(end:-1:2)];
end

% Reshape H to broadcast along DIM only
sh = ones(1, ndims(Xpad));
sh(dim) = L;
Hfull = reshape(Hfull, sh);

% FFT-based filtering
Ypad = ifft(fft(Xpad, [], dim) .* Hfull, [], dim, 'symmetric');

% Remove the reflection, keep the original length along DIM
idx = repmat({':'}, 1, ndims(Ypad));
idx{dim} = 1:n;
Y = Ypad(idx{:});

% Cast back to original class if needed (preserve integers where safe)
if ~isa(Y, origClass)
    switch origClass
        case {'single'}
            Y = single(Y);
        case {'double'}
            % already double
        otherwise
            % For integer types, return double to avoid unintended clipping.
            % Users can cast explicitly if needed.
    end
end

end
