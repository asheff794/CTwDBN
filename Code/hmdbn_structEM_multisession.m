function [seeddag, hiddenGraph_Ps, SampleDistribution] = hmdbn_structEM_multisession(TimeSeriesDataCell, seed)

fprintf('\n');
fprintf('##########################################################################################\n');
fprintf('########   Multisession Hidden Markov induced Dynamic Bayesian Network (msHMDBN)  ########\n');
fprintf('########                         by Alec Sheffield                                ########\n');
fprintf('########              modified from code written by Shijia Zhu                    ########\n');
fprintf('##########################################################################################\n');
fprintf('\n');



% Number of sessions
num_sessions = length(TimeSeriesDataCell);

% Initialize variables
data_sessions = cell(num_sessions, 1);
ncases_sessions = zeros(num_sessions, 1);
ns_sessions = cell(num_sessions, 1);

for s = 1:num_sessions
    TimeSeriesData = TimeSeriesDataCell{s};
    [ns, ts]=size(TimeSeriesData);

    data=zeros(2*ns,(ts-1));
    data(1:ns,1:(ts-1))=(TimeSeriesData(:,1:(ts-1)));
    data((ns+1):(2*ns),1:(ts-1))=(TimeSeriesData(:,2:(ts)));
    data_sessions{s} = data;

    ns_sessions{s} = max(data, [], 2)'; % Compute the number of states for each variable

    [N, ncases] = size(data);
    ncases_sessions(s) = ncases;

end

% % Compute overall ns (number of states per variable) by taking the max over sessions
% ns = zeros(1, N);
% for s = 1:num_sessions
%     ns = max(ns, ns_sessions{s});
% end

hiddenGraph_Ps = cell(N,1); % Shared across sessions
SampleDistribution = cell(num_sessions, N); % Per session

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%  initial values for transition matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_fan_in = N;
NG = 2;
init_state_distrib = ones(1, NG) / NG;
total_ncases = sum(ncases_sessions);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  posterior probability for hmdbn with two nodes
%  that is gamma in forward and backward algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('==> Calculating initial distribution Pr(qt|X,HMDBN)...\n');

allgamma_sessions=cell(num_sessions,N,N);

for s=1:num_sessions
    data = data_sessions{s};
    ncases = ncases_sessions(s);
    ns = ns_sessions{s};

    ot=(1/ncases)/(NG-1);
    transmat=diag((1-2*ot)*ones(1,NG),0)+ot;

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
            [~, ~, allgamma_sessions{s,i,j}, ~, ~] = hmdbn_fwdback(init_state_distrib, transmat, obslik);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  construct random starting graph   %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%           Alec Addition            %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(seed,'twister')
seeddag=zeros(N,N);
for i = 1:(N/2)
    for j = (N/2+1):N
        seeddag(i,j) = round(rand(1));
    end
end
disp('Initial Seed DAG:');
disp(seeddag);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%  initial scores for hmdbn without edges   %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('==> Calculating initial BWBIC score...\n');

% allscore_sessions = cell(num_sessions, 1);
% for s = 1:num_sessions
%     allscore_sessions{s} = zeros(1, N);
% end

allscore=zeros(1,N);
for j=(1):N
    ps = parents(seeddag,j);
    for s=1:num_sessions
        data = data_sessions{s};
        ns = ns_sessions{s};
        allgamma = squeeze(allgamma_sessions(s,:,:));

        [score, ~, ~]= hmdbn_hiddenGraphs_and_BWBIC_node(j, ps, ns,  data, allgamma);
        allscore(j) =  allscore(j) + score * ncases_sessions(s) / total_ncases;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Alec is working on this part


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%   greedy climing to learn hmdbn   %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('==> Reconstructing the time-evolving DBN...\n');

outcount=0;
while outcount < 2
    innercount = 0;

    for i=1:(N/2)

        fprintf('  ==> scanning Node%d...\n',i);

        for j=(N/2+1):N

            % if i==j, continue;    end; %This line prevents self-edges

            if (i+N/2)==j, continue;    end
            if seeddag(i,j) == 0  % No edge i-->j, then try to add it
                tempdag = seeddag;
                tempdag(i,j)=1;
                ps=parents(tempdag,j);
                
                % [temp_score, temp_gamma, temp_hiddenGraph_Ps]= hmdbn_hiddenGraphs_and_BWBIC_node(j, ps, ns,  data,allgamma);
                temp_score = 0;
                for s=1:num_sessions
                    data = data_sessions{s};
                    ns = ns_sessions{s};
                    allgamma = squeeze(allgamma_sessions(s,:,:));
            
                    [temp_score, temp_gamma, temp_hiddenGraph_Ps] = hmdbn_hiddenGraphs_and_BWBIC_node(j, ps, ns,  data, allgamma);
                    temp_score =  temp_score + score * ncases_sessions(s) / total_ncases;
                end

                if temp_score > allscore(j)  &&  sum( seeddag(:,j) ) < max_fan_in

                    seeddag = tempdag;
                    allscore(j)= temp_score;
                    SampleDistribution{j} = temp_gamma;
                    hiddenGraph_Ps{j} = temp_hiddenGraph_Ps;
                    innercount =innercount+1;

                    fprintf('    ( + ) add edge: Node%d-->Node%d\n',i,j-N/2);

                end

            else  % exists edge i--j, then try to remove it
                tempdag = seeddag;
                tempdag(i,j) = 0;
                ps=parents(tempdag,j);
                [temp_score, temp_gamma, temp_hiddenGraph_Ps]= hmdbn_hiddenGraphs_and_BWBIC_node(j, ps, ns,  data, allgamma);

                if temp_score > allscore(j)

                    fprintf('    ( - ) remove edge: Node%d-->Node%d\n',i,j-N/2);

                    innercount = innercount+1;
                    seeddag = tempdag;
                    allscore(j)= temp_score;
                    SampleDistribution{j} = temp_gamma;
                    hiddenGraph_Ps{j} = temp_hiddenGraph_Ps;

                end

            end


        end % end for j
    end % end for i

    if innercount == 0
        outcount = outcount +1;
    end

end  % end while
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end % end function

