function [A,mto_score,results,desc,P,SDDM,conf_matrix]=runAllLanguages(attDict,r_1,k)

langDir='../data/';
%langs = {'ptb17','ptb45','pt','bg','es','dk'};
langs = {'ptb17'};
desc=0;

nrK = 1;
nrLangs = size(langs,2)
results=zeros(nrK,nrLangs)
for i=1:nrK
  for lnr = 1:nrLangs
    l = langs(lnr)
    l{1}
    corpus = load ([langDir l{1} '/' l{1} '-sentences.all.word']);
	r_1 = 50; 
    k = 100*i;
    sigma_schedule = eval('.2*exp(-r_1*.01*(0:14))');
    mce = 1;
    tags = load ([langDir l{1} '/' l{1} '-sentences.all.pos']);
    dictFile = [langDir l{1} '/' l{1} '-ldc.pred']  
    scoreFile = [langDir l{1} '/' l{1} '-ldc.score']  
    
    %if svdorcca==1
    	[A, mto_score,desc,P,SDDM,conf_matrix] = ldclustering(corpus, k, sigma_schedule, r_1, mce, tags);
    %else
	%[A, mto_score,P,SDDM,conf_matrix,desc] = ldclusteringCCA(corpus,attDict, k, sigma_schedule, r_1, mce, tags);
    %end	
    L_induced = A(corpus);              % induced labelling of corpus
    dlmwrite(dictFile, A);
    fileId = fopen(scoreFile,'w');
    fprintf(fileId, '%s %s', 'Many to one: ',num2str(mto_score(1:1)));
    mto_score
    results(i,lnr)=mto_score;
  end   
end
