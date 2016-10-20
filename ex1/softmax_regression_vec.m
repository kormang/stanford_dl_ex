function [f,g] = softmax_regression(theta, X, y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%

etX = [exp(theta' * X); ones(1, m)]; % K x m
etX_colsum = sum(etX); % 1 x m
Ps = bsxfun(@rdivide, etX, etX_colsum); % K x m

index = sub2ind(size(Ps), y, 1:m);

P = Ps(index); % 1 x m
logP = log(P); % 1 x m

f = - sum(logP); % 1 x 1

y01 = zeros(size(Ps)); % K x m
y01(index) = 1; % K x m

g = - X * (y01 - Ps)'; % n x K

g=g(:, 1:end - 1); % n x (K - 1)

g=g(:); % make gradient a vector for minFunc
