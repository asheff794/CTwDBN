function hmdbn_demonstrate(TimeSeriesData,hiddenGraph_Ps,SampleDistribution,range)

ncase = size(TimeSeriesData,2);
[N n]=size(hiddenGraph_Ps);
if nargin < 4, range = [1:N/2]; end

figure(length(range));


for g1=1:length(range)

    subplot( length(range),1,g1)
    sampleD = SampleDistribution{ range(g1)+ N/2 }';
    
    if (size(sampleD,2)== 0)
        
		yl = sprintf(  '%s%d%s','Pr(qt|X,HMDBN' , range(g1) , '):                              ' );
		ylabel( yl ,'rot',0)
        axis([0, ncase, 0, 1])
        
    elseif (size(sampleD,2)>0)
        
        set( 0,'DefaultAxesColorOrder',jet( size(sampleD,2) ))
        %x = linspace(0,500);
        plot(sampleD,'LineWidth',3);
        h =legend(cellstr(num2str((1:size(sampleD,2))')));
        v = get(h,'title');

        names=cell(size(hiddenGraph_Ps{  range(g1)+N/2  },2),1);
        for g2 =1:size(hiddenGraph_Ps{  range(g1)+N/2  },2)
            names{g2} = mat2str(cell2mat(hiddenGraph_Ps{  range(g1)+N/2  }(1,g2)));
        end

        legend(names)
		yl = sprintf(  '%s%d%s','Pr(qt|X,HMDBN' , range(g1) , '):                              ' );
		ylabel( yl ,'rot',0)
        axis([0, ncase, 0, 1])
      
    end
end

%title('')
xlabel('Time Points')
%
        
