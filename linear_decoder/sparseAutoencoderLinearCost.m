function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:2*hiddenSize*visibleSize+hiddenSize+visibleSize);

%forward prop:
z1 = bsxfun(@plus, W1 * data, b1);
a1 = sigmoid(z1);

m = size(data, 2);
ro = sparsityParam;
ro_hats = mean(a1, 2);
g1 = ro ./ ro_hats;
g2 = (1 - ro)./(1 - ro_hats);
sparsityCost = beta * sum(ro * log(g1) + (1 - ro) * log(g2));

z2 = bsxfun(@plus, W2 * a1, b2);
errors = z2 - data;

errorCost = 1/(2*m) * sum(sum(errors .* errors));

regularizationCost = lambda/2 * (sum(sum(W1 .* W1)) + sum(sum(W2 .* W2)));

a1deriv = a1 .* (1 - a1);
gradW2 = 1/m * (errors * a1');
gradb2 = 1/m * sum(errors, 2);

gradW2_reg = lambda * W2;
gradW1_reg = lambda * W1;

delta_sparse = bsxfun(@times, 1/m * a1deriv, g2 - g1);
gradW1_sparse = beta * (delta_sparse * data');
gradb1_sparse = beta * sum(delta_sparse, 2);

cost = sparsityCost + errorCost + regularizationCost;


delta = W2' * errors;
delta = delta .* a1deriv;

% release some resources
a1deriv = [];
delta_sparse = [];
z2 = [];
errors = [];


gradW1 = 1/m * (delta * data');
gradb1 = 1/m * sum(delta, 2);

gradW2 = gradW2 + gradW2_reg;

gradW1 = gradW1 + gradW1_sparse + gradW1_reg;
gradb1 = gradb1 + gradb1_sparse;

grad = [gradW1(:); gradW2(:); gradb1; gradb2];

end


function output = sigmoid(input)
    output = 1 ./ (1 + exp(-input));
end