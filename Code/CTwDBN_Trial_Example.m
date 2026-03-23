%% Path Setup
% Project layout assumed: CTwDBN/{Code,Data}. Run from repo root.

if exist('Code','dir') ~= 7 || exist('Data','dir') ~= 7
	error('CTwDBN:Path', 'Run from the repository root so Code/ and Data/ exist.');
end

addpath('Code');

dataDir = 'Data';
outFile = fullfile('Data', 'continuous_dags_all.mat');

%% Generate Synthetic Data (Trial-Based, multiple trials)
% % Specify how many trials to generate; each trial uses a different seed.
% % Results are saved under Data/ by simulate_struct_bernoulli_bin_3d.
% 
% num_trials = 10000; 
% 
% % One-call trial generation and saving (no plotting)
% [data_cells, graph_cells, prob_cells] = simulate_struct_bernoulli_bin_3d(num_trials);

%% Fit CTwDBN on all trials (collect continuous DAGs only)
% Parfor to speed up across trials. Requires Parallel Computing Toolbox.

% % Ensure Code is on path (client)
% addpath('Code');
% 
% if ~exist(dataDir, 'dir')
% 	error('CTwDBN_Trial_Example:DataMissing', 'Data directory not found at %s. Generate trials first.', dataDir);
% end
% 
% % Precompute file list
% trialFiles = cell(1, num_trials);
% for k = 1:num_trials
% 	trialFiles{k} = fullfile(dataDir, sprintf('bernoulli_trial_%03d.mat', k));
% end
% 
% % Preallocate results container
% continuous_dags_all = cell(1, num_trials);
% 
% % Use a parpool if not already started
% try
% 	gcp('nocreate');
% catch
% 	% If Parallel Toolbox is missing, fallback to serial for-loop
% end
% 
% parfor k = 1:num_trials
% 	% Each worker ensures Code is on path
% 	addpath('Code');
% 
% 	S = load(trialFiles{k}, 'd');
% 	TimeSeriesData = S.d;        % (nodes x T), saved by simulate_*
% 	TimeSeriesData = TimeSeriesData + 1;  % convert 0/1 -> 1/2 for HMDBN indexing
% 
% 	seed = k - 1;                % deterministic seed per trial
% 	verbose = false;             % quiet
% 	condition = sprintf('trial_%03d', k);
% 
% 	[~, ~, ~, ~, continuous_dags] = ctwdbn_structEM(TimeSeriesData, seed, verbose, condition);
% 	continuous_dags_all{k} = continuous_dags;
% end
% 
% save(outFile, 'continuous_dags_all', '-v7.3');

%% Visualize: CTwDBN weights vs. ground truth (window 1000:2000)
% Average prediction over trials, compare to ground-truth probs


S = load(outFile, 'continuous_dags_all');
A = S.continuous_dags_all;          % 1 x num_trials cell, each (T x 3 x 3)
num_trials = numel(A);

[T, n, ~] = size(A{1});
pred_sum = zeros(T, n, n);
for k = 1:num_trials
	pred_sum = pred_sum + double(A{k});
end
pred = pred_sum / num_trials;       % (T x n x n)

G = load(fullfile(dataDir, 'bernoulli_trial_001.mat'), 'p');
gt = G.p;                           % (T x n x n)

tStart = 1000; tEnd = 2000;
pred_win = pred(tStart:tEnd, :, :);
gt_win   = gt(tStart:tEnd,   :, :);
t = (1:(tEnd - tStart + 1))';

tl = tiledlayout(3,3, 'TileSpacing','compact', 'Padding','compact');
h_pred = [];
h_gt = [];
for i = 1:n  % from-node (row)
	for j = 1:n  % to-node (col)
		nexttile((i-1)*n + j);
		if i == j
			axis off; continue;
		end

		yp = pred_win(:, i, j);
		yg = gt_win(:,   i, j);
		yp = (yp - min(yp)) / max(eps, (max(yp) - min(yp)));
		yg = (yg - min(yg)) / max(eps, (max(yg) - min(yg)));

		yyaxis left; hp = plot(t, yp, 'Color', [11 119 52]/255, 'LineWidth', 1.2); set(gca,'YColor',[11 119 52]/255);
		yyaxis right; hg = plot(t, yg, '--k', 'LineWidth', 1.0); set(gca,'YColor',[0 0 0]);
		title(sprintf('%d → %d', i, j), 'FontSize', 10);

		% Hide ticks by default; only show on bottom-left
		if i == 3 && j == 1
			xlabel('Time');
			ylabel('Pred.');
			yyaxis right; ylabel('GT'); yyaxis left;
		else
			set(gca, 'XTickLabel', []);
			yyaxis left; set(gca, 'YTickLabel', []);
			yyaxis right; set(gca, 'YTickLabel', []);
		end

		box off;
		if isempty(h_pred)
			h_pred = hp; h_gt = hg;
		end
	end
end

legend([h_pred h_gt], {'Prediction','Ground truth'}, 'Location', 'northoutside', 'Orientation', 'horizontal', 'Box', 'off');

