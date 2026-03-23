function [alpha, beta, gamma, xi,transmat] = hmdbn_updateA(gamma,xi,transmat,NG,obslik,init_state_distrib,ns,seeddagall,data)

% this function iteratively update transition matrix
dt=1;
num=1;
while num<100 & dt
    num=num+1;
    [priorNew, transmatNew] =hmdbn_mhmm_A(gamma,xi);  
    dt=sum(sum(log(transmatNew+0.00001).*xi))-sum(sum(log(transmat+0.00001).*xi));                        
    transmat=transmatNew;                        
	[alpha, beta, gamma,loglik,xi] = hmdbn_fwdback(init_state_distrib, transmat, obslik);
end
