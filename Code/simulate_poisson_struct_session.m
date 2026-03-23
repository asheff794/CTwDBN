function [data, states, transition_points, state_series] = simulate_poisson_struct_session( ...
    total_duration, alpha, lambda_initial, epsilon, rng_seed, state_duration_range)
% One-session generator (MATLAB port of your Python).
% Returns:
%   data  : [6 x total_duration] integer counts (n_populations is fixed at 6)
%   states: [1 x n_states] sequence of state IDs in {1,2,3,4}
%   transition_points: [1 x (n_states+1)] 1-based sample indices where each state starts;
%                      the last element marks the end index of the final state.
%   state_series: [1 x total_duration] per-timepoint state label in {1,2,3,4}
%
% Defaults (adjust freely)
if nargin < 1 || isempty(total_duration), total_duration = 30*60*1000; end   % 1,800,000 steps
if nargin < 2 || isempty(alpha),          alpha          = 0.1;         end
if nargin < 3 || isempty(lambda_initial), lambda_initial = 0.1;         end
if nargin < 4 || isempty(epsilon),        epsilon        = 0.01;        end
if nargin >= 5 && ~isempty(rng_seed), rng(rng_seed); end
if nargin < 6 || isempty(state_duration_range), state_duration_range = [30000 300000]; end

% Fixed number of populations for tutorial (hard-coded connections)
n_populations = 6;

% Validate state duration range [min max]
validateattributes(state_duration_range, {'numeric'}, {'vector','numel',2,'finite','nonnegative'});
state_duration_range = double(state_duration_range(:)');
if state_duration_range(1) > state_duration_range(2)
    state_duration_range = fliplr(state_duration_range);
end
% Ensure minimum is at least 1 sample
state_duration_range(1) = max(1, floor(state_duration_range(1)));
state_duration_range(2) = max(state_duration_range(1), floor(state_duration_range(2)));

T = total_duration;

% ---------------- Dependency sets (source, target, lag, beta), 1-based ----------------
% State 1
deps{1} = [ ...
    1 2 2  0.5;   % 1E -> 1I
    2 1 2 -0.5;   % 1I -> 1E
    3 4 2  0.5;   % 2E -> 2I
    4 3 2 -0.5;   % 2I -> 2E
    5 6 2  0.5;   % 3E -> 3I
    6 5 2 -0.5;   % 3I -> 3E
    1 3 3  0.5];  % 1E -> 2E

% State 2
deps{2} = [ ...
    1 2 2  0.5;
    2 1 2 -0.5;
    3 4 2  0.5;
    4 3 2 -0.5;
    5 6 2  0.5;
    6 5 2 -0.5;
    5 3 3  0.5];  % 3E -> 2E

% State 3
deps{3} = [ ...
    1 2 2  0.5;
    2 1 2 -0.5;
    3 4 2  0.5;
    4 3 2 -0.5;
    5 6 2  0.5;
    6 5 2 -0.5;
    1 3 3  0.5;   % 1E -> 2E
    3 5 3  0.5;   % 2E -> 3E
    5 1 3  0.5];  % 3E -> 1E

% State 4
deps{4} = [ ...
    1 2 2  0.5;
    2 1 2 -0.5;
    3 4 2  0.5;
    4 3 2 -0.5;
    5 6 2  0.5;
    6 5 2 -0.5;
    5 3 3  0.5;   % 3E -> 2E
    3 1 3  0.5;   % 2E -> 1E
    1 5 3  0.5];  % 1E -> 3E

% ---------------- Initialization ----------------
data  = poissrnd(lambda_initial, n_populations, T);
states = [];
transition_points = [];
state_series = zeros(1, T);   % fill per block with 1..4

current_time = 1;                 % 1-based index (inclusive)
state_ids = 1:4;                  % states are 1..4 (and depsets indexed by 1..4)

% ---------------- Main loop: sample piecewise-constant states ----------------
while current_time <= T
    % Pick next state (different from previous if any)
    if isempty(states)
        state = state_ids(randi(numel(state_ids)));
    else
        candidates = setdiff(state_ids, states(end));
        state = candidates(randi(numel(candidates)));
    end

    % Duration sampled from user-provided range [min max] (inclusive)
    duration = randi([state_duration_range(1), state_duration_range(2)]);

    % Trim to remaining time
    if current_time + duration - 1 > T
        duration = T - current_time + 1;
    end

    % Record state and transition start
    states(end+1) = state; %#ok<AGROW>
    transition_points(end+1) = current_time; %#ok<AGROW>

    % Generate samples for this block
    end_idx = generate_samples_block(state, duration, data, current_time, deps, epsilon, alpha);
    % Fill per-timepoint state labels
    state_series(current_time:end_idx) = state;  % state already in 1..4

    % Advance
    current_time = end_idx + 1;
end

% Mark the end of the last state
transition_points(end+1) = T;

% ---------------- Nested: evolve data within one state block ----------------
    function end_idx = generate_samples_block(st, dur, data_ref, start_idx, depsets, eps_val, a)
        D = depsets{st};                    % rows: [src tgt lag beta], src/tgt are 1-based
        lags = unique(D(:,3));
        maxLag = max(lags);
        end_idx = start_idx + dur - 1;

        % For times before start_idx+maxLag, we rely on the already-initialized data_ref
        for t = (start_idx + maxLag) : end_idx
            % update each population
            for target = 1:n_populations
                lambda_t = a * data_ref(target, t-1);

                % add dependency terms that target this population
                % (vectorized over rows matching target)
                rows = (D(:,2) == target);
                if any(rows)
                    srcs = D(rows, 1);
                    lgs  = D(rows, 3);
                    bet  = D(rows, 4);

                    past_vals = data_ref(sub2ind(size(data_ref), srcs, t - lgs));
                    lambda_t = lambda_t + sum(bet .* past_vals);
                end

                % floor
                lambda_t = max(lambda_t, eps_val);

                % sample
                data_ref(target, t) = poissrnd(lambda_t);
            end
        end

        % write back
        data(:, start_idx:end_idx) = data_ref(:, start_idx:end_idx);
    end
end
