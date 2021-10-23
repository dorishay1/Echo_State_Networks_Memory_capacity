function [Y0] = Y0_calc(input,output_size)
%calculating Y0 according to formula.

k = output_size;

%the loop starts to save only when there are valid Y0 answers (t>k)
for t = 1:length(input)
    if t>k
    Y0(:,t-k) = input((t-1):-1:(t-k));
    end
end

Y0 = Y0';
end

