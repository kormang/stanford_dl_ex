%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
epsilon = params.epsilon;
lambda = params.lambda;

Wx = W * x;
WTWx = W' * Wx;
WTWxmx = WTWx - x;

L2normWTWxmx = L2norm(WTWxmx);
reconstructionTerm = 0.5 * L2normWTWxmx .^ 2;

sqrtWx2peps = sqrt(Wx .^ 2 + epsilon);
L1normWx = sum(sum(sqrtWx2peps));
sparsityCost = lambda * L1normWx;

cost = sparsityCost  + reconstructionTerm;

delta = 2 * WTWxmx;
grad_WT = delta * Wx';
delta2 = W * delta;
grad_W = delta2 * x';
gradReconstruction = 0.5 * (grad_WT' + grad_W);

gradSparsity = lambda * (Wx ./ sqrtWx2peps) * x';

Wgrad = gradSparsity + gradReconstruction;


% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);

function [output] = L2norm(input)
  output = sqrt(sum(sum(input .* input)));
