function [R] = ff_ESN(input,hidden_size,output_size,time_steps,alpha,W_in,K,act_func)
%feedforward for ESN
r = zeros(1,hidden_size);

for t = 1:time_steps
    X = input(t);
    r = (1 - alpha) * r + alpha * act_func(X * W_in + r * K);
    
    %storing r with bias and input neurons.
    if t>output_size
        R_temp(:,t-output_size) = [1, X, r];
    end
end

R = R_temp;
end

