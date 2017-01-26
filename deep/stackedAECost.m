function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example

close all
addpath(genpath('..'))

%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
%softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
% for d = 1:numel(stack)
%     stackgrad{d}.W = zeros(size(stack{d}.W));
%     stackgrad{d}.b = zeros(size(stack{d}.b));
% end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.W and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%


% forward propagate:

regularizationCost = 0;

acts = cell(size(stack));

layerInputs = data;

for level = 1:numel(stack)
   regularizationCost = regularizationCost + sum(sum(stack{level}.W .* stack{level}.W));
   layerInputs = sigmoid(bsxfun(@plus, stack{level}.W * layerInputs, stack{level}.b));
   acts{level} = layerInputs;
end

etX = exp(softmaxTheta * layerInputs);
etX_colsum = sum(etX);

Ps = bsxfun(@rdivide, etX, etX_colsum);

Pindex = sub2ind(size(Ps), labels', 1:(size(Ps, 2)));
P = Ps(Pindex);
logP = log(P);

cost = -sum(logP) + 0.5 * lambda * regularizationCost;

delta = (Ps - groundTruth);

softmaxThetaGrad = delta * layerInputs';
delta = (softmaxTheta' * delta) .* layerInputs .* (1 - layerInputs);

for level = numel(stack):-1:1
    stackgrad{level}.b = sum(delta, 2);
    if level > 1
        stackgrad{level}.W = (delta * acts{level - 1}') + lambda * stack{level}.W; % including regularization term
        delta = (stack{level}.W' * delta) .* acts{level - 1} .* (1 - acts{level - 1});
    end
end
stackgrad{1}.W = (delta * data')  + lambda * stack{1}.W; % including regularization term;

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

assert(isequal(size(grad), size(theta)), 'Oh no!')
end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
