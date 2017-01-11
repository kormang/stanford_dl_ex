function [Z, U, S, V] = zca2(x)
epsilon = 1e-1;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

%%% YOUR CODE HERE %%%

avg = mean(x, 1);
x = bsxfun(@minus, x, avg);
sigma = x * x' * (1 / size(x, 2));
[U, S, V] = svd(sigma);
xRot = U' * x;
xPCAWhite = diag(1 ./ (diag(S) + epsilon)) * xRot; % make it have unit variance
Z = U * xPCAWhite;

