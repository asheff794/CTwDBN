function [score tempCPD]= hmdbn_BWBIC_score(j, ps, ns,  data,gammaG)

% this function is the previous hmdbn_score_family()
% score = hmdbn_BWBIC_node(j, ps, node_type, scoring_fn, ns, discrete, data, args, cache)

% this function calculates the BWBIC score for a specific emission graph for j, where ps are the parent nodes of j (ps -> j)
% data(i,m) is the value of node i in case m 

        ps = unique(ps);
        fam = [ps j];
        
        tempCPD = hmdbn_learn_params(fam,ns, data(:,:), gammaG);
        %L = log_prob_node(bnet.CPD{j}, data(j,:), data(ps,:));
        L = hmdbn_log_prob_node(tempCPD, data(j,:), data(ps,:),gammaG);
        S = prod(size(tempCPD))/ns(j); % violate object privacy
        ncases=sum(gammaG)+1e-4;
        score = L - 0.5*S*log(ncases);
    

