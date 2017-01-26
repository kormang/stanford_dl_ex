function [output] = feedForwardAutoencoder(theta, hiddenSize, inputSize, input)

W = reshape(theta(1: hiddenSize*inputSize), hiddenSize, inputSize);
b = theta(2*hiddenSize*inputSize + 1 : 2*hiddenSize*inputSize + hiddenSize);

output = sigmoid(bsxfun(@plus, W * input, b));

end
