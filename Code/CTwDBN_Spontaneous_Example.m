%% Path Setup
% Project layout assumed: CTwDBN/{Code,Data}. Run from repo root.

if exist('Code','dir') ~= 7 || exist('Data','dir') ~= 7
	error('CTwDBN:Path', 'Run from the repository root so Code/ and Data/ exist.');
end
addpath('Code');

%% Spontaneous tutorial: generate one session
% Parameters
% T = 30 * 60 * 1000;  % 30 minutes in 1 ms time steps
% alpha = 0.1;         % AR coefficient
% lambda0 = 0.1;       % base rate
% epsilon = 0.01;      % floor
% rng_seed = 42;        % for reproducibility
% state_range = [30000 300000];    % state durations (inclusive)
% 
% % Generate one session
% [data, states, transition_points, state_series] = simulate_poisson_struct_session( ...
% 	T, alpha, lambda0, epsilon, rng_seed, state_range);
% 
% % Save
% outfile = fullfile('Data', 'Spontaneous_Session_001.mat');
% save(outfile, 'data', 'states', 'transition_points', 'state_series', 'T', 'alpha', 'lambda0', 'epsilon', 'rng_seed', 'state_range');
% fprintf('Saved spontaneous session -> %s\n', outfile);

%% Fit CTwDBN on the session (50 seeds, parallel)
% Load session
% S = load(fullfile('Data', 'Spontaneous_Session_001.mat'), 'data');
% X = S.data;                     % integer counts (6 x T)
% 
% % Preprocess: combine E/I pairs -> 3 populations, then sliding window rebin (width=3)
% X_comb = [X(1,:)+X(2,:); X(3,:)+X(4,:); X(5,:)+X(6,:)];   % 3 x T
% X_rebin = conv2(X_comb, ones(1,3), 'same');               % zero-padded, same length
% 
% TimeSeriesData = X_rebin + 1; %HMDBN treats 1 as no events/spikes
% 
% % Seeds
% seeds = 1:50;
% N = numel(seeds);
% continuous_dags_seeds = cell(1, N);
% 
% % Parallel fit across seeds
% tic
% parfor idx = 1:N
% 	addpath('Code');
% 	sd = seeds(idx);
% 	condition = sprintf('spontaneous_s1_seed%02d', sd);
% 	[~, ~, ~, ~, continuous_dags] = ctwdbn_structEM(TimeSeriesData, sd, false, condition);
% 	continuous_dags_seeds{idx} = continuous_dags;
% end
% toc
% save(fullfile('Data', 'Spontaneous_continuous_dags_seeds.mat'), 'continuous_dags_seeds', 'seeds', '-v7.3');
% fprintf('Saved continuous DAGs (50 seeds) -> %s\n', fullfile('Data', 'Spontaneous_continuous_dags_seeds.mat'));

%% Average across seeds, then smooth with Gaussian
S2 = load(fullfile('Data','Spontaneous_continuous_dags_seeds.mat'), 'continuous_dags_seeds', 'seeds');
C = S2.continuous_dags_seeds;   % 1 x N cell, each (T x n x n)
N = numel(C);

sigma = 30000/4;            % set sigma to 1/4 of shortest state

% Prepare accumulation directly in vectorized edge space (T x n^2)
Tlen = size(C{1},1);
n = size(C{1},2);
accM = zeros(Tlen, n*n, 'double');

for k = 1:N
	D = double(C{k});                % (T x n x n)
	M = reshape(D, [Tlen, n*n]);     % columns are edges over time
	accM = accM + M;                 % sum before smoothing
end

avgM = accM / N;                      % average across seeds first
% Apply zero-phase Gaussian smoothing once to the averaged series
Xk = fft_gauss_reflect(avgM, sigma, 1);  % (T x n^2)
save(fullfile('Data','Spontaneous_edge_features.mat'), 'Xk', 'sigma', '-v7.3');
fprintf('Saved smoothed average edge features -> %s\n', fullfile('Data','Spontaneous_edge_features.mat'));

%% Bootstrap K-means (100 repeats) on vectorized edge features (T x n^2)
% Uses Xk from the previous step (T x n^2), observations are timepoints
Tlen = size(Xk,1);

% Bootstrap settings
klist = 2:10;                  % cluster counts to evaluate
n_repeats = 100;               % number of bootstrap iterations
sample_size = min(1000, Tlen); % rows sampled per bootstrap

% Storage for all repeats (rows = repeats, cols = k values)
nk = numel(klist);
inertia_all = zeros(n_repeats, nk);
sil_all = zeros(n_repeats, nk);

% Parallel bootstrap over repeats; resample with replacement per k
parfor b = 1:n_repeats
	% Deterministic per-iteration stream for reproducibility
	stream = RandStream('CombRecursive','Seed', 12345 + b);
	local_inertia = zeros(1, nk);
	local_sil = zeros(1, nk);

	for ii = 1:nk
		k = klist(ii);
		% Bootstrap sample indices with replacement
		idx_bs = randi(stream, Tlen, sample_size, 1);
		Xs = Xk(idx_bs, :);

		% K-means (squared Euclidean inertia; 10 replicates with kmeans++)
		[idx, ~, sumd] = kmeans(Xs, k, 'Replicates', 10, 'MaxIter', 200, ...
								'Distance','sqeuclidean', 'Start','plus', 'Display','off');
		local_inertia(ii) = sum(sumd);

		% Silhouette with Euclidean distances (suppress plot)
		fh = figure('Visible','off');
		s = silhouette(Xs, idx, 'euclidean');
		local_sil(ii) = mean(s);
		close(fh);
	end

	inertia_all(b, :) = local_inertia;
	sil_all(b, :) = local_sil;
end

% Aggregate: mean and SEM over bootstraps
inertia_mean = mean(inertia_all, 1);
inertia_sem  = std(inertia_all, 0, 1) / sqrt(n_repeats);
sil_mean     = mean(sil_all, 1);
sil_sem      = std(sil_all, 0, 1) / sqrt(n_repeats);

% Plot mean ± SEM for inertia and silhouette vs k
figure;
subplot(1,2,1);
errorbar(klist, inertia_mean, inertia_sem, '-o'); grid on;
xlabel('k'); ylabel('Inertia'); title('K-means inertia (mean ± SEM)');
subplot(1,2,2);
errorbar(klist, sil_mean, sil_sem, '-o'); grid on;
xlabel('k'); ylabel('Mean silhouette'); title('K-means silhouette (mean ± SEM)');

%% K-means on full dataset (K=4)
% Perform K-means on all timepoints to obtain stable labels and centroids
rng(42, 'twister');
[km4_labels, km4_centroids] = kmeans(Xk, 4, 'Replicates', 10, 'MaxIter', 300, ...
	'Distance','sqeuclidean', 'Start','plus', 'Display','off');

% Persist results for reuse in later scripts/figures
save(fullfile('Data','KMeans4_full_Xk.mat'), 'km4_labels', 'km4_centroids', '-v7.3');
fprintf('Saved K=4 K-means labels and centroids -> %s\n', fullfile('Data','KMeans4_full_Xk.mat'));

% Map K-means labels to ground-truth states (1..4) for consistent colors
Gmap = load(fullfile('Data','Spontaneous_Session_001.mat'), 'state_series');
st_full_map = Gmap.state_series(:);
K = 4;
Tuse = min(numel(st_full_map), size(Xk,1));
gt_use = st_full_map(1:Tuse);
km_use = km4_labels(1:Tuse);

% Confusion matrix C(state, cluster)
Cmap = zeros(K, K);
for s = 1:K
	for c = 1:K
		Cmap(s,c) = sum(gt_use == s & km_use == c);
	end
end

% Find best permutation of cluster->state maximizing agreement
P = perms(1:K);
best_score = -Inf; best_p = 1:K;
for i = 1:size(P,1)
	p = P(i,:); % p(c) = state assigned to cluster c
	score = sum(arrayfun(@(c) Cmap(p(c), c), 1:K));
	if score > best_score
		best_score = score;
		best_p = p;
	end
end

% Apply mapping to labels: mapped label equals ground-truth state index
km4_labels_mapped = arrayfun(@(c) best_p(c), km4_labels);

% Reorder centroids so row s corresponds to mapped state s
km4_centroids_mapped = zeros(size(km4_centroids));
for c = 1:K
	s = best_p(c);
	km4_centroids_mapped(s, :) = km4_centroids(c, :);
end

% Save mapped variants to disk for downstream use
save(fullfile('Data','KMeans4_full_Xk.mat'), 'km4_labels_mapped', 'km4_centroids_mapped', '-append');

%% PCA on vectorized DAG features (first two PCs), colored by ground-truth state
% Xk is (T x n^2) with timepoints as rows
[coeff, score, latent, tsquared, explained] = pca(Xk);

G = load(fullfile('Data','Spontaneous_Session_001.mat'), 'state_series');
st = G.state_series(:);   % states in 1..4

% Align lengths if needed
m = min(size(score,1), numel(st));
score = score(1:m, :);
st = st(1:m);

% Subsample a random set of timepoints for visualization
nsamp = min(10000, m);
idx_sample = randperm(m, nsamp);
score_s = score(idx_sample, :);
st_s = st(idx_sample);

cols = [0.12 0.47 0.71;  % group 1
	1.00 0.50 0.05;  % group 2
	0.20 0.63 0.17;  % group 3
	0.89 0.10 0.11]; % group 4

figure;
% Left: colored by ground-truth state
subplot(1,2,1); hold on;
hs = gobjects(1,4);
for s = 1:4
    idx_gt = (st_s == s);
    hs(s) = scatter(score_s(idx_gt,1), score_s(idx_gt,2), 50, cols(s,:), 'filled', ...
	'MarkerFaceAlpha', 0.02, 'MarkerEdgeAlpha', 0.2);
end
grid on; xlabel('PC 1'); ylabel('PC 2'); title('PCA colored by ground-truth state');
legend(hs, {'State 1','State 2','State 3','State 4'}, 'Location','bestoutside');

% Right: colored by K-means (K=4) labels mapped to GT states
subplot(1,2,2); hold on;
hc = gobjects(1,4);
labels_s = km4_labels_mapped(1:m);
labels_s = labels_s(idx_sample);
for c = 1:4
    idxc = (labels_s == c);
    hc(c) = scatter(score_s(idxc,1), score_s(idxc,2), 50, cols(c,:), 'filled', ...
	'MarkerFaceAlpha', 0.02, 'MarkerEdgeAlpha', 0.2);
end
grid on; xlabel('PC 1'); ylabel('PC 2'); title('PCA colored by K-means (K=4)');
legend(hc, {'Cluster 1','Cluster 2','Cluster 3','Cluster 4'}, 'Location','bestoutside');

%% Compare timing: ground-truth state transitions vs K-means (K=4)
% Compute transition indices
st_full = load(fullfile('Data','Spontaneous_Session_001.mat'), 'state_series');
st_full = st_full.state_series(:);
Tlen = size(Xk,1);

tp_gt = find(diff(st_full(1:Tlen)) ~= 0) + 1;     % ground-truth transitions
tp_km = find(diff(km4_labels_mapped(1:Tlen)) ~= 0) + 1;  % K-means (mapped) label transitions

% Use the same 4-color palette
cols_vis = [0.12 0.47 0.71;  1.00 0.50 0.05;  0.20 0.63 0.17;  0.89 0.10 0.11];

figure;
subplot(2,1,1);
imagesc(1:Tlen, 1, st_full(1:Tlen)');
colormap(gca, cols_vis); caxis([1 4]);
set(gca,'YTick',[]); xlim([1 Tlen]);
title('Ground truth states'); ylabel('');
hold on; if ~isempty(tp_gt), xline(tp_gt, 'k:'); end; hold off;

subplot(2,1,2);
imagesc(1:Tlen, 1, km4_labels_mapped(1:Tlen)');
colormap(gca, cols_vis); caxis([1 4]);
set(gca,'YTick',[]); xlim([1 Tlen]);
title('K-means states (K=4)'); xlabel('Time'); ylabel('');
hold on; if ~isempty(tp_km), xline(tp_km, 'k:'); end; hold off;

%% Centroids (3x3) vs ground-truth connectivity (3x3) for each state
% Define ground-truth directly as 3x3 binary adjacencies (1 = connection),
% consistent with consolidated populations (E/I already summed).

gt_adj = zeros(3,3,4);
% State 1: 1 -> 2
gt_adj(:,:,1) = [0 1 0; 0 0 0; 0 0 0];
% State 2: 3 -> 2
gt_adj(:,:,2) = [0 0 0; 0 0 0; 0 1 0];
% State 3: 1 -> 2, 2 -> 3, 3 -> 1
gt_adj(:,:,3) = [0 1 0; 0 0 1; 1 0 0];
% State 4: 1 -> 3, 2 -> 1, 3 -> 2
gt_adj(:,:,4) = [0 0 1; 1 0 0; 0 1 0];

% Prepare centroid matrices (mapped order), zero diagonals for display
cent_adj = zeros(3,3,4);
for s = 1:4
	C = reshape(km4_centroids_mapped(s,:), [3,3]);
	C(1:4:end) = 0;
	cent_adj(:,:,s) = C;
end

% Also create a thresholded (binary) version of centroids: > 0.4 -> 1, else 0
thr_adj = cent_adj > 0.4;

% Plot: row 1 GT, row 2 centroids, row 3 thresholded centroids (states 1..4 across columns)
figure;
tiledlayout(3,4, 'TileSpacing','compact', 'Padding','compact');
for s = 1:4
	nexttile(s);
	imagesc(gt_adj(:,:,s), [0 1]); axis image off;
	title(sprintf('GT state %d', s));
	colormap(gca, parula);
end
for s = 1:4
	nexttile(4 + s);
	imagesc(cent_adj(:,:,s), [0 1]); axis image off;
	title(sprintf('Centroid %d', s));
	colormap(gca, parula);
end
for s = 1:4
	nexttile(8 + s);
	imagesc(double(thr_adj(:,:,s)), [0 1]); axis image off;
	title(sprintf('Centroid %d (>0.4)', s));
	colormap(gca, parula);
end
cb = colorbar('Location','eastoutside');
cb.Layout.Tile = 'east';
sgtitle('Ground-truth vs K-means centroids (3x3) and thresholded (>0.4)');
