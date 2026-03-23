function [seeddag hiddenGraph_Ps SampleDistribution allscore continuous_dags] = ctwdbn_structEM(TimeSeriesData, seed, print, condition)
% CTwDBN_STRUCTEM Continuous Time weighted Dynamic Bayesian Network structure learning
%
% This function learns the structure of a Hidden Markov induced Dynamic Bayesian Network
% and returns both the traditional outputs and a continuously evolving DAG representation.
%
% Inputs:
%   TimeSeriesData - Time series data matrix (nodes x time_points)
%   seed - Random seed for initialization
%   print - Boolean flag for verbose output
%   condition - String identifier for checkpoint files
%
% Outputs:
%   seeddag - Final DAG structure matrix
%   hiddenGraph_Ps - Hidden graph parent sets for each node
%   SampleDistribution - Posterior probabilities over time for parent sets
%   allscore - BWBIC scores for each node
%   continuous_dags - 3D array (time x from_nodes x to_nodes) representing continuously
%               evolving DAG where edge weights are accumulated probabilities
%
% The continuous_dags output is equivalent to the Python build_dag function output,
% eliminating the need for post-processing in Python.

if print
    fprintf('\n');
    fprintf('############################################################################\n');
    fprintf('########  Continuous Time weighted Dynamic Bayesian Network (CTwDBN) #######\n');
    fprintf('########                   by Alec Sheffield; based on HMDBN by          ###\n');
    fprintf('########                             original author: Shijia Zhu         ###\n');
    fprintf('############################################################################\n');
    fprintf('\n');
end

[ns, ts]=size(TimeSeriesData);
data=zeros(2*ns,(ts-1));
data(1:ns,1:(ts-1))=(TimeSeriesData(:,1:(ts-1)));
data((ns+1):(2*ns),1:(ts-1))=(TimeSeriesData(:,2:(ts)));
[N, ncases] = size(data);


hiddenGraph_Ps = cell(N,1);
SampleDistribution = cell(N,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%  initial values for transition matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if nargin < 2, max_fan_in = N; end
max_fan_in = N;

% ns=2*ones(1,N); %This line only allows binary variables
ns = max(data, [], 2)'; % Compute the number of states for each variable

NG=2;
init_state_distrib=ones(1,NG)/NG;
ot=(1/ncases)/(NG-1);
transmat=diag((1-2*ot)*ones(1,NG),0)+ot;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  posterior probability for hmdbn with two nodes
%  that is gamma in forward and backward algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if print
    fprintf('==> Calculating initial distribution Pr(qt|X,HMDBN)...\n');
end

allgamma=cell(N,N);
gamma=zeros(2,ncases)+0.5;
obslik=zeros(NG,ncases);

for i=1:(N/2)
    for j=(N/2+1):N

        tempCPDnode=cell(1,2);
        for gi=1:NG
            if(gi==1)
                ps=i;
            end
            if(gi==2)
                ps=[];
            end
            fam = [ps j];
            tempCPDnode{gi} = hmdbn_learn_params(fam,ns, data(:,:), gamma);
        end
        for gi=1:NG
            if(gi==1)
                ps=i;
            end
            if(gi==2)
                ps=[];
            end

            self_ev=data(j,:);
            pev=data(ps,:);
            CPD= tempCPDnode{gi};
            obslik(gi,:) = hmdbn_prob_node(CPD,self_ev, pev);
        end
    [~, ~, allgamma{i,j}, ~, ~] = hmdbn_fwdback(init_state_distrib, transmat, obslik);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  construct random starting graph   %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(seed,'twister')
seeddag=zeros(N,N);
for i = 1:(N/2)
    for j = (N/2+1):N
        seeddag(i,j) = round(rand(1));
    end
end
if print
    disp('Initial Seed DAG:');
    disp(seeddag);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%  initial scores for hmdbn without edges   %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if print
    fprintf('==> Calculating initial BWBIC score...\n');
end
allscore=zeros(1,N);
for j=(1):N
    ps = parents(seeddag,j);
    [allscore(j), ~, ~] = hmdbn_hiddenGraphs_and_BWBIC_node(j, ps, ns,  data, allgamma);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%   greedy climing to learn hmdbn   %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if print
    fprintf('==> Reconstructing the time-evolving DBN...\n');
end
outcount=0;

complete = 0;
search_idx = 0;
% Determine default Checkpoints directory relative to this file (repo root)
thisFile = mfilename('fullpath');
thisDir = fileparts(thisFile);
repoRoot = fileparts(thisDir); % Code/ under repo root
ckptDir = fullfile(repoRoot, 'Checkpoints');
if ~exist(ckptDir, 'dir')
    mkdir(ckptDir);
end
filename = fullfile(ckptDir, sprintf('%s_checkpoint%d.mat', condition, search_idx));
save(filename, 'seeddag', 'complete');

while outcount < 2
    innercount = 0;

    for i=1:(N/2)
        if print
            fprintf('  ==> scanning Node%d...\n',i);
        end
        for j=(N/2+1):N

            % if i==j, continue;    end; %This line prevents self-edges
            if (i+N/2)==j, continue;    end
            if seeddag(i,j) == 0  % No edge i-->j, then try to add it
                tempdag = seeddag;
                tempdag(i,j)=1;
                ps=parents(tempdag,j);
                [temp_score, temp_gamma, temp_hiddenGraph_Ps]= hmdbn_hiddenGraphs_and_BWBIC_node(j, ps, ns,  data,allgamma);

                if temp_score > allscore(j)  &&  sum( seeddag(:,j) ) < max_fan_in
                    if print
                        fprintf('    ( + ) add edge: Node%d-->Node%d\n',i,j-N/2);
                    end

                    innercount =innercount+1;
                    seeddag = tempdag;
                    allscore(j)= temp_score;
                    SampleDistribution{j} = temp_gamma;
                    hiddenGraph_Ps{j} = temp_hiddenGraph_Ps;
                    
                    search_idx = search_idx + 1;
                    filename = fullfile(ckptDir, sprintf('%s_checkpoint%d.mat', condition, search_idx));
                    save(filename, 'seeddag', 'complete');


                end

            else  % exists edge i--j, then try to remove it
                tempdag = seeddag;
                tempdag(i,j) = 0;
                ps=parents(tempdag,j);
                [temp_score, temp_gamma, temp_hiddenGraph_Ps]= hmdbn_hiddenGraphs_and_BWBIC_node(j, ps, ns,  data,allgamma);

                if temp_score > allscore(j)
                    if print
                        fprintf('    ( - ) remove edge: Node%d-->Node%d\n',i,j-N/2);
                    end

                    innercount = innercount+1;
                    seeddag = tempdag;
                    allscore(j)= temp_score;
                    SampleDistribution{j} = temp_gamma;
                    hiddenGraph_Ps{j} = temp_hiddenGraph_Ps;
                    
                    search_idx = search_idx + 1;
                    filename = fullfile(ckptDir, sprintf('%s_checkpoint%d.mat', condition, search_idx));
                    save(filename, 'seeddag', 'complete');

                end

            end


        end % end for j
    end % end for i

    if innercount == 0
        outcount = outcount +1;
    end

end  % end while

complete = 1;
save(filename, 'seeddag', 'complete');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Build continuously evolving DAG %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize DAG
continuous_dags = zeros(ncases, N/2, N/2);

% Build the DAG
for child = (N/2+1):N
    child_idx = child - N/2;
    
    if isempty(SampleDistribution{child})
        continue;
    end
    
    parent_probs = SampleDistribution{child};
    potential_parents = hiddenGraph_Ps{child};
    
    for g = 1:size(parent_probs, 1)
        parent_nodes = potential_parents{g};
        weights = parent_probs(g, :);
        
        % Make parents a vector if it's a scalar
        if isscalar(parent_nodes)
            parent_nodes = [parent_nodes];
        end
        
        % Add weights for each parent
        for p = parent_nodes
            continuous_dags(:, p, child_idx) = continuous_dags(:, p, child_idx) + weights(:);
        end
    end
end

if print
    fprintf('==> Continuously evolving DAG construction complete.\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end % end function

