function L = hmdbn_log_prob_node(CPD, self_ev, pev,gammaG)
% LOG_PROB_NODE Compute sum_m log P(x(i,m)| x(pi_i,m), theta_i) for node i (discrete)
% L = hmdbn_log_prob_node(CPD, self_ev, pev,gammaG)
%
% self_ev is the evidence on this node.
% pev(i,) is the evidence on the i'th parent

% Herein, the probability is the product of gammaG and log(p) by hmdbn_prob_node().
% gammaG is the probability P(O, g=G| HMDBNi), that is the probability that one sample
% belongs to the graph G.

p = hmdbn_prob_node(CPD, self_ev, pev); % P may underflow, so we use p
tiny = exp(-700);
p = p + (p==0)*tiny; % replace 0s by tiny
pg=log(p)'.*gammaG;
L = sum(pg);
