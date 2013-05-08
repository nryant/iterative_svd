function []=mtoscore(file)

corpus = load ('../data/ptb17/ptb17-sentences.all.word');
tags = load ('../data/ptb17/ptb17-sentences.all.pos');

L_induced = file(corpus); 
conf_matrix = sparse(tags, L_induced, 1, 17, 50);
[label_MTO_scores, ~] = max(conf_matrix);
mto_score = sum(label_MTO_scores)/length(corpus)
