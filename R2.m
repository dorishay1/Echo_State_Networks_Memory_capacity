function [result] = R2(Y,Y0,output_size)
%calculate R^2

for k = 1:output_size
    result(k,1) = corr(Y(:,k),Y0(:,k)).^2;
end

end

