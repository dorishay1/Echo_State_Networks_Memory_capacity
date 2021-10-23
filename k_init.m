function [K] = k_init(hidden_size,connectivity,delta)
%Caluclating K matrix

%random weights
temp_K = randn(hidden_size);

%connectivity
conn_mat = rand(hidden_size);
conn_mat = conn_mat < connectivity;

temp_K = temp_K.*conn_mat;

%regulate K
K = (1/(max(real(eig(temp_K))))).*(temp_K.*delta);

end

