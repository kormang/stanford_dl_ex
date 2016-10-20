% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

pkg load statistics

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

num_iter = 10;
delta=1e-4;
sum_error=0;

for i = 1:num_iter

data_train = rand(4, 10);
labels_train = floor(rand(10, 1) * 2) + 1;

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 4;
% number of output classes
ei.output_dim = 2;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [2, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0.3;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

T = params;
param_index = randsample(numel(T), 1);
T0 = T; T0(param_index) = T0(param_index) - delta;
T1 = T; T1(param_index) = T1(param_index) + delta;

[fval, grad, ~] = supervised_dnn_cost(T, ei, data_train, labels_train);
fval0 = supervised_dnn_cost(T0, ei, data_train, labels_train);
fval1 = supervised_dnn_cost(T1, ei, data_train, labels_train);
g_est = (fval1 - fval0) / (2*delta);
err = abs(grad(param_index) - g_est);
fprintf('pidx = %d, grad(pidx) = %g, f0 = %g, f1 = %g, g_est = %g, err = %g\n',
	param_index, grad(param_index), fval0, fval1, g_est, err);

sum_error = sum_error + err;
end

fprintf('Avarage error = %g\n', sum_error / num_iter);

%% compute accuracy on the test and train set
%[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
%[~,pred] = max(pred);
%acc_test = mean(pred'==labels_test);
%fprintf('test accuracy: %f\n', acc_test);

%[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
%[~,pred] = max(pred);
%acc_train = mean(pred'==labels_train);
%fprintf('train accuracy: %f\n', acc_train);
