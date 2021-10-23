function [g, gp] = softmax_act(x,y0)
%this function returns the softmax solution and its derivative
%for two classes:(g_y = g(y0==1))
%
%g = softmax(x)
%
%if y == i
%gp = g_y*(1-g_y)
%
%else
%gp = -g_y*g_i

    
    gp = zeros(size(x));
    y_logic = ligical(y0);
    g = softmax(x);
    gp(y_logic) = g(logical(y0)).*(1-g(logical(y0)));
    gp(~y_logic) = -g(y_logic).*(g(~y_logic));    
end