function [score SampleDistribution hiddenGraph_g2_Ps]= hmdbn_hiddenGraphs_and_BWBIC_node(j, ps, ns,  data,SeedSampleDistribution)
% this is the previous function hmdbn_score_family_all_diffgammanologm()
% [score tempCPD]= hmdbn_score_family_all_diffgammanologm(j, ps, ns,  data,SeedSampleDistribution)
% SeedSampleDistribution is the p(O, G|hmdbn), where hmdbn is the nonstationary network with only two networks composed of two nodes i and j.
% The two networks, one is only one node j and the other network is node (i and j) and edge i -> j.

% 1. this function identifies the putative hidden graphs according to SeedSampleDistribution;
% 2. according to SeedSampleDistribution,set the initial p(O, G|hmdbn) for each putative hidden graph;
% 3. function hmdbn_fwdback() further modifies the transition matrix; (specific putative hidden graphs may be removed)
% 4. acorrding to the above results (transition matrix and hidden graphs), calculte the BWBIC score.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% identifie the putative hidden graphs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[N n]=size(data);

StationaryPsNum=length(ps);
SeedGraphPath=zeros(StationaryPsNum,n);
for pi=1:StationaryPsNum
    SeedGraphPath(pi,:)=round(SeedSampleDistribution{ps(pi),j}(1,:));
end

sz = ns(ps);
P = prod(sz);
hiddenGraphPath = subv2ind(sz, SeedGraphPath'+1);
hiddenGraphType=unique(hiddenGraphPath);
hiddenGraphNum=length(hiddenGraphType);

SampleDistribution=ones(hiddenGraphNum,n);
hiddenGraph_g2_Ps=cell(1,hiddenGraphNum);
hiddenGraph=cell(1,hiddenGraphNum);


if hiddenGraphNum==0  
    
    [score tempCPD]= hmdbn_BWBIC_score(j, ps, ns,  data,ones(1,n));
    
else 

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% set the initial p(O, G|hmdbn) for each putative hidden graph
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        for g2=1:hiddenGraphNum
               range=find(hiddenGraphPath==hiddenGraphType(g2)); 
               hiddenGraph_g2_Ps_Index=SeedGraphPath(:,range(1))';
               hiddenGraph_g2_Ps{g2}=hiddenGraph_g2_Ps_Index.*ps;
               hiddenGraph_g2_Ps{g2}=ps(find(hiddenGraph_g2_Ps{g2}~=0));
               hiddenGraph_g2_Ps_Index2=2-hiddenGraph_g2_Ps_Index;
			   hiddenGraph{g2}=zeros(N,N);
               hiddenGraph{g2}(hiddenGraph_g2_Ps{g2},j)=1;
			   
               for pi=1:StationaryPsNum
                  SampleDistribution(g2,:)=SampleDistribution(g2,:).*SeedSampleDistribution{ps(pi),j}(hiddenGraph_g2_Ps_Index2(pi),:);
               end
        end
        SampleDistribution=hmdbn_normalise(SampleDistribution,1);
		
        obsliknode=zeros(hiddenGraphNum,n);
		
        for g2=1:hiddenGraphNum

               fam=[hiddenGraph_g2_Ps{g2} j];
               %ps{ge} = unique(ps{ge});
               gammaG=SampleDistribution(g2,:);
               CPD = hmdbn_learn_params(fam,ns, data(:,:), gammaG);
               self_ev=data(j,:);
               pev=data(hiddenGraph_g2_Ps{g2},:);
               obsliknode(g2,:) = hmdbn_prob_node(CPD,self_ev, pev);

        end
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%		
		%%     further modify the transition matrix  
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Pi=ones(1,hiddenGraphNum)/hiddenGraphNum;
        ot=(1/n);%/(NG-1);
        transmat=diag((1-2*ot)*ones(1,hiddenGraphNum),0)+ot;%A;
        [alpha, beta, SampleDistribution,loglik,xi] = hmdbn_fwdback(Pi, transmat, obsliknode);
        [alpha, beta, SampleDistribution, xi,transmat] = hmdbn_updateA(SampleDistribution,xi,transmat,hiddenGraphNum,obsliknode,Pi,ns,hiddenGraph,data);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%%%%%%%  BWBIC score	%%%%%%%%%%%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		score=0;
        for g2=1:hiddenGraphNum     
            gammaG=SampleDistribution(g2,:);
            score = score + hmdbn_BWBIC_score(j, hiddenGraph_g2_Ps{g2}, ns,  data,gammaG); 
        end
end

