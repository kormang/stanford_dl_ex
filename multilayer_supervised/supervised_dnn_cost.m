function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+2, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%

z = [];
a = data;

for i = 1:numHidden
  z = bsxfun(@plus, stack{i}.W * a, stack{i}.b);
  a = f_act(z);
  hAct{i} = a;
end

z = bsxfun(@plus, stack{numHidden + 1}.W * a, stack{numHidden + 1}.b);
etX = [exp(z)];
etX_colsum = sum(etX);
Ps = bsxfun(@rdivide, etX, etX_colsum);
hAct{numHidden + 1} = Ps;

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  pred_prob = Ps;  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
Pindex = sub2ind(size(Ps), labels', 1:(size(Ps, 2)));
P = Ps(Pindex);
logP = log(P);
cost = -sum(logP);

wCost = 0;

%% perform regularization
for i = 1:numHidden + 1
  wCost = wCost + sum(sum(stack{i}.W .* stack{i}.W));
end

cost = cost + 0.5 * ei.lambda * wCost;

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
labels01 = zeros(size(Ps));
labels01(Pindex) = 1;
%d = -sum(labels01 - Ps, 2);
d = Ps - labels01;

for i = numHidden+1:-1:1
  gradStack{i} = struct;
  gradStack{i}.b = sum(d, 2);
  if i > 1
    gradStack{i}.W = d * (hAct{i - 1})';
    d = (stack{i}.W' * d) .* (hAct{i - 1} .* (1 - hAct{i - 1}));
  end
end
gradStack{1}.W = d * data';

%% include regularization terms

for i = 1:numHidden + 1
  gradStack{i}.W = gradStack{i}.W + ei.lambda * stack{i}.W;
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%


%% reshape gradients into vector
[grad] = stack2params(gradStack);

end



