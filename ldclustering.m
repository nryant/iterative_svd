function [A, mto_score,desc,P,SDDM,conf_matrix] = ldclustering(corpus, k, sigma_schedule, r_1, mce, tags)
%LDCLUSTERING     Latent-Descriptor Clustering (LDC)
%   LDCLUSTERING(CORPUS, K, SIGMA_SCHEDULE, R_1, MCE, TAGS) performs LDC of CORPUS
%   for POS tagging.
%   The LDC algorithm is described in
%   Latent-Descriptor Clustering for Unsupervised POS Induction,
%   Michael Lamar, Yariv Maron, and Elie Bienenstock, EMNLP 2010,
%   referred to below as "the paper."
%   In the present application, LDC is used for fully-unsupervised,
%   distributional-only, non-disambiguating POS tagging.
%   Non-disambiguating means that the algorithm clusters word types rather than tokens.

%   Data file CORPUS should be a row vector of length N_tokens,
%   where N_tokens is the length of the corpus, and CORPUS(i) is the word type
%   of the token that appears in the i-th position in the corpus, suitably encoded.
%   Specifically, letting N_types be the number of word types that appear in the corpus,
%   CORPUS(i) should be an integer between 1 and N_types,
%   and the encoding should be by decreasing frequency.
%   This means that the most frequent word type in the corpus should be encoded as 1,
%   the second most frequent word type as 2, etc.
%   
%   Parameter K is the number of labels to induce.
%   Parameter SIGMA_SCHEDULE should be a row vector of decreasing positive real numbers,
%   typically between 0.5 and 0.
%   The number of iterations is equal to LENGTH(SIGMA_SCHEDULE).
%   Parameter R_1, the reduced rank for iteration 1, should be a positive integer,
%   typically equal to or less than K.
%   Parameter MCE is a flag that determines whether or not to use mixture coefficients.
%   Set MCE to 0 to NOT use mixture coefficients (equivalent to setting all coefficients to 1/K).
%   Set MCE to 1 to use mixture coefficients.
%   The version of LDC described in detail in the paper does NOT use mixture coefficients.
%   This gives the best results under MTO, which is the most reliable criterion.
%   It appears that under the OTO criterion best results are obtained 
%   by using mixture coefficients, i.e., setting parameter MCE to 1.

%   TAGS (gold-standard tagging used for evaluation) should be a row vector of length N_tokens,
%   and TAGS(i) should be the gold-standard tag
%   of the token that appears in the i-th position in the corpus, suitably encoded.
%   Specifically, letting N_tags be the number of tags used in the gold-standard tagging,
%   TAGS(i) should be an integer between 1 and N_tags.
%   Gold-standard tags may be encoded in any order.

%   The algorithm returns:
%   A, the clustering, i.e., an assignment of each of the N_types word types to one of the K labels;
%   MTO_SCORE, the accuracy of the tagging of the corpus under the Many-to-One mapping.

%   Intermediate results, including the accuracy under the One-to-One mapping
%   and some indicators of convergence, are printed out
%   in the command window for each iteration.

%   Examples of use of LDCLUSTERING, 
%   with suggested parameter settings for the WSJ section of the PTB corpus,
%   can be found in WSJ_LDC_POS_TAGGER.

%   Function ASSIGNMENTOPTIMAL (by Markus Buehren) is used to construct the one-to-one mapping.

%   The present code, as well as WSJ_LDC_POS_TAGGER and ASSIGNMENTOPTIMAL
%   are available from http://www.dam.brown.edu/people/elie/code/

%   Elie Bienenstock
%   Last modified September 29, 2010

%% A FEW USEFUL THINGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_tokens = length(corpus);                                          % number of tokens, i.e., corpus length
N_types  = length(unique(corpus));                                  % number of word types, i.e., vocabulary size
N_tags   = length(unique(tags));                                    % number of gold-standard tags

f = hist(corpus,1:N_types)'/N_tokens;                               % word-type frequencies

R = sparse(corpus(1:end-1), corpus(2:end), 1, N_types, N_types);    % right-word context matrix
L = R';                                                             %  left-word context matrix


% FUNCTION DEFINITIONS

LAMBDA = @(X) repmat(1./sqrt(sum(X.^2,2)),1,size(X,2)).*X;
%               scales to unit length (using L2 norm) each row of input matrix X

vect_dist_sq = @(X,Y) repmat(sum(X.^2,2), 1, size(Y,1)) + repmat(sum(Y.^2,2), 1, size(X,1)).' - 2*X*Y.';
%               Input X,Y: two matrices of row vectors, with same number of columns, possibly different numbers of rows
%               Output Z: Z(i,j) is ||X(i,:)-Y(j,:)||^2


% Initializing some arrays:
length_schedule = length(sigma_schedule);
MTO_scores      = zeros(1, length_schedule);
G               = zeros(1, length_schedule);    % objective function

disp(' ')
tic

%% LDC ALGORITHM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for ite = 1:length_schedule
    
    sig = sigma_schedule(ite);
    
    %%%%%%%   "M STEP" OF LDC ALGORITHM   %%%%%%%
    
    if ite == 1          
       
        % we will use the word context, SVD-ed down to rank R_1        
        [U, S, ~] = svds(R, r_1);               
        R_P = LAMBDA(U*S);                          % right part of latent descriptors
        [U, S, ~] = svds(L, r_1);          
        L_P  = LAMBDA(U*S);                         %  left part of latent descriptors
        clear U S
        
    else                                            % we will use the tag context,
                                                    %   derived from last iteration's assignments (P)       
        R_P = LAMBDA(R*P);
        L_P = LAMBDA(L*P);
    end
    
    
    latent_descriptors = [L_P R_P];% full latent descriptors
    desc=latent_descriptors;
    desc(1,:)
    clear L_P R_P
        
    if ite == 1                  
        
        MU = latent_descriptors(1:k, :);            % Gaussian means are descriptors of K most frequent words                            
        mix_coeff = ones(1, k)/k;                   % mixture coefficients, if used, are uniform
                                                    
    else                                 
        
        P_f = P.*repmat(f, 1, size(P,2));           % assignments weighted by word-type frequencies  
        MU = P_f'*latent_descriptors;               % Gaussian means use last iteration's assignments
                                                    %   and newly-computed latent descriptors
                                         
        H = size(MU, 2)/2;                          % length of each of the left/right descriptor vectors 

        MU = [LAMBDA(MU(:,1:H)) LAMBDA(MU(:,1+H:2*H))]; % scale each half separately to unit length
                                                    
        if mce                                      % use mixture coefficients
            mix_coeff = mean(P);                    % mixture coefficients
        else                                        % don't use mixture coefficients
            mix_coeff = ones(1, k)/k;               %   i.e., make them uniform
        end

    end
    
    %%%%%%%   "E STEP" OF LDC ALGORITHM   %%%%%%%

    SDDM = vect_dist_sq(latent_descriptors,MU);     % squared distances between all descriptors and all Gaussian means
    x = SDDM/(2*sig^2);                             % scale by 2*sig^2
    x = x - repmat(min(x,[],2), 1, k);              % subtract minimum from each row to avoid underflow when exponentiating
    x = exp(-x);                                    % exponentiate
    x = x.*repmat(mix_coeff, N_types, 1);           % multiply by mixture coefficients
    P = x./repmat(sum(x,2), 1, k);                  % normalize each row to get probabilistic assignments
    
    
    %% EVALUATION OF TAGGING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [~,A] = max(P, [], 2);                                  % hard assignments
    
    if ite > 1
        G(ite) = sum(sum(P_f.*SDDM))/2;                     % objective function
    end

    nnc = sum((P(:)>0) & (P(:)<1));                         % number of P entries not yet converged to 0 or 1
        
    L_induced = A(corpus);                                  % induced corpus labeling
    
    
    conf_matrix = sparse(tags, L_induced, 1, N_tags, k);    % confusion matrix between gold-standard and induced taggings
    
    [label_MTO_scores, ~] = max(conf_matrix);                                   % MTO score for each label
    mto_score = sum(label_MTO_scores)/N_tokens;                                 % total MTO score
    MTO_scores(ite) = mto_score;
   % [~, OTO_score] = assignmentoptimal(max(conf_matrix(:)) - conf_matrix);      % negative total OTO score + constant
   % OTO = (min(size(conf_matrix))*max(conf_matrix(:)) - OTO_score)/N_tokens;    % total OTO score
  %  disp([' it=' num2str(ite) ' sig=' num2str(sig) ' MTO=' num2str(mto_score) ' OTO=' num2str(OTO) ' G=' num2str(G(ite)) ' NNC=' num2str(nnc)])
    disp([' it=' num2str(ite) ' sig=' num2str(sig) ' MTO=' num2str(mto_score) ' G=' num2str(G(ite)) ' NNC=' num2str(nnc)])
    
end

toc

%% DISPLAY RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot(1:length_schedule, MTO_scores, 'k*-', 'linewidth',2, 'markersize',14)
%ylim([0 max(MTO_scores)+.01])
%set(gca,'fontsize',16)
%hold on
%plot(2:length_schedule, G(2:end), 'ko-', 'linewidth',1.5, 'markersize',14)
%plot(1:length_schedule, sigma_schedule, 'k+-', 'linewidth',1.5, 'markersize',14)
%title([num2str(N_tags) ' tags    ' num2str(k) ' labels'])
%legend('MTO', '{\it G}', '{\it \sigma}', 'location','best')
%xlim([0 length_schedule])
%set(gca,'XTick',0:5:length_schedule)
%grid on
%xlabel('iteration', 'fontsize', 16)
%hold off
