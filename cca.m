function [CCARWm1W,CCARWp1W,CCALWm1W,CCALWp1W]= cca(corpus)

N_tokens = length(corpus);                                          % number of tokens, i.e., corpus length
N_types  = length(unique(corpus));

W=sparse(1:N_tokens, corpus(1:end),1, N_tokens, N_types);
Wm1=sparse(2:N_tokens,corpus(1:end-1),1, N_tokens, N_types);
Wp1=sparse(1:N_tokens-1,corpus(1:end-1),1, N_tokens, N_types);

%invWW=inv(W'*W);
%invWm1Wm1=inv(Wm1'*Wm1);
%invWp1Wp1=inv(Wp1'*Wp1);
Crl_m=(W'*Wm1);
Clr_m=Crl_m';

Crl_p=(W'*Wp1);
Clr_p=Crl_p';
right=Crl_m*Clr_m;
[U,S]= eigs(right,50);
CCARWm1W=U*S;
[U,S]= eigs(Crl_p*Clr_p,50);
CCARWp1W=U*S;

left=Clr_m*Crl_m;
[U,S]= eigs(left,50);
CCALWm1W=U*S;
[U,S]= eigs(Clr_p*Crl_p,50);
CCALWp1W=U*S;

