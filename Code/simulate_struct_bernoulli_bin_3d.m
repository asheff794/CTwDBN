function [data, graphs, probs] = simulate_struct_bernoulli_bin_3d(varargin)
% simulate_struct_bernoulli_bin_3d(num_timepoints, buffer, period, base_p, amplitude,
%                                  bias, self_weight, cross_weight, random_seed)
%
% If only one input is provided, it is interpreted as
% the NUMBER OF TRIALS to generate. Each trial uses a different seed (0..N-1),
% is saved to Data/ as bernoulli_trial_XXX.mat, and outputs are also returned as
% cell arrays {1..N} for [data, graphs, probs].

% ---------------- handle inputs ----------------
if nargin == 1
    % One-arg mode: interpret the single input as number of trials
    num_trials = varargin{1};
    validateattributes(num_trials, {'numeric'}, {'scalar','finite','integer','>=',1});

    % Resolve Data directory relative to this file (repoRoot/Data)
    thisFile = mfilename('fullpath');
    thisDir = fileparts(thisFile);
    repoRoot = fileparts(thisDir); % src/ under root
    dataDir = fullfile(repoRoot, 'Data');
    if ~exist(dataDir, 'dir')
        mkdir(dataDir);
    end

    fprintf('simulate_struct_bernoulli_bin_3d: generating %d trials -> %s\n', num_trials, dataDir);

    % Preallocate cell outputs
    data = cell(1, num_trials);
    graphs = cell(1, num_trials);
    probs = cell(1, num_trials);

    % Use defaults for other parameters and vary only the seed
    for trialIdx = 1:num_trials
        seed = trialIdx - 1; % deterministic different seed per trial
        [d, g, p] = simulate_struct_bernoulli_bin_3d(3000, 1000, 1000, 0.5, 0.35, -2.0, -1, 2, seed);
        data{trialIdx} = d;
        graphs{trialIdx} = g;
        probs{trialIdx} = p;

        outPath = fullfile(dataDir, sprintf('bernoulli_trial_%03d.mat', trialIdx));
        save(outPath, 'd', 'g', 'p', 'seed');
        fprintf('  saved trial %d (seed=%d) -> %s\n', trialIdx, seed, outPath);
    end

    % Early return in trials mode
    return;
else
    % Fill in missing inputs with defaults
    defaults = {3000, 1000, 1000, 0.5, 0.35, -2.0, -1, 2, []};
    for k = 1:nargin
        defaults{k} = varargin{k};
    end
    [num_timepoints, buffer, period, base_p, amplitude, ...
        bias, self_weight, cross_weight, random_seed] = defaults{:};
end

% ---------------- RNG setup ----------------
if ~isempty(random_seed)
    rng(random_seed);
end

% ---------------- main body ----------------
n = 3;
T = num_timepoints;
omega = 2*pi / double(period);

L = zeros(n, n, 3);
L(1,2,:) = [ 1; 0; 0];
L(2,3,:) = [ 0; 1; 0];
L(3,1,:) = [ 0; 0; 1];
L(2,1,:) = [-1; 0; 0];
L(3,2,:) = [ 0; -1; 0];
L(1,3,:) = [ 0; 0; -1];

probs = zeros(T, n, n);

for t = 1:T
    tt = t - 1;
    core_t = min(max(double(tt), double(buffer)), double(T - buffer - 1));
    vt_buf = [sin(omega*core_t); cos(omega*core_t); sin(2*omega*core_t)];
    for i = 1:n
        for j = 1:n
            if i == j
                probs(t,i,j) = 1.0;
            else
                lij = squeeze(L(i,j,:)).';
                p = base_p + amplitude * (lij * vt_buf);
                probs(t,i,j) = min(max(p,0),1);
            end
        end
    end
end

graphs = double(rand(T,n,n) < probs);
for t = 1:T
    graphs(t,:,:) = squeeze(graphs(t,:,:)) | eye(n);
end

data = zeros(n, T);
data(:,1) = rand(n,1) < 0.5;
sigmoid = @(x) 1./(1+exp(-x));

for t = 2:T
    for j = 1:n
        lin = bias;
        for i = 1:n
            if graphs(t,i,j)
                lin = lin + (i==j)*self_weight*data(i,t-1) + (i~=j)*cross_weight*data(i,t-1);
            end
        end
        data(j,t) = rand < sigmoid(lin);
    end
end
end
