function [output] = f_act(input)

output = 1 ./ (1 + exp(-input));

end
