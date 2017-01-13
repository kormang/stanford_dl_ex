%% CS294A/CS294W Self-taught Learning Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises.
%
%% ======================================================================
%  STEP 0: Here we provide the relevant parameters values that will
%  allow your RICA to get good filters; you do not need to 
%  change the parameters below.
close all
addpath(genpath('..'))
imgSize = 28;
global params;
params.patchWidth=9;           % width of a patch
params.n=params.patchWidth^2;   % dimensionality of input to RICA
params.lambda = 0.0001;   % regularization cost
params.beta = 3;   % sparsity cost
params.ro = 0.005;   % sparsity param
params.numFeatures = 32; % number of filter banks to learn
params.epsilon = 1e-2;

%% ======================================================================
%  STEP 1: Load data from the MNIST database
%
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.

% Load MNIST database files
mnistData   = loadMNISTImages('../common/train-images-idx3-ubyte');
mnistLabels = loadMNISTLabels('../common/train-labels-idx1-ubyte');

numExamples = size(mnistData, 2);
% 50000 of the data are pretended to be unlabelled
unlabeledSet = 1:50000;
unlabeledData = mnistData(:, unlabeledSet);

% the rest are equally splitted into labelled train and test data


trainSet = 50001:55000;
testSet = 55001:60000;
trainData   = mnistData(:, trainSet);
trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-10
% only keep digits 0-4, so that unlabelled dataset has different distribution
% than the labelled one.
removeSet = find(trainLabels > 5);
trainData(:,removeSet)= [] ;
trainLabels(removeSet) = [];

testData   = mnistData(:, testSet);
testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-10
% only keep digits 0-4
removeSet = find(testLabels > 5);
testData(:,removeSet)= [] ;
testLabels(removeSet) = [];


% Output Some Statistics
fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in supervised training set: %d\n\n', size(trainData, 2));
fprintf('# examples in supervised testing set: %d\n\n', size(testData, 2));

%% ======================================================================
%  STEP 2: Train the RICA
%  This trains the RICA on the unlabeled training images. 

%  Randomly initialize the parameters
params.hiddenSize = params.numFeatures;
params.visibleSize = params.n;
hiddenSize = params.hiddenSize;
visibleSize = params.visibleSize;
randTheta = initializeSAEParameters(params.hiddenSize, params.visibleSize);
randTheta = randTheta(:);

% subsample random patches from the unlabelled+training data
patches = samplePatches([unlabeledData,trainData],params.patchWidth,200000);

%configure minFunc
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 1000;
% You'll need to replace this line with RICA training code
%opttheta = randTheta;

%  Find opttheta by running the RICA on all the training patches.
%  You will need to whitened the patches with the zca2 function 
%  then call minFunc with the softICACost function as seen in the RICA exercise.
%%% YOUR CODE HERE %%%

x = normalizeDataForSAE(patches);

% display('checking SAE gradients');
% numgrad = computeNumericalGradient(@(theta) sparseAutoencoderCost(theta, params.visibleSize, params.hiddenSize, params.lambda, params.ro, params.beta, x), randTheta);
% [~, grad] = sparseAutoencoderCost(randTheta, params.visibleSize, params.hiddenSize, params.lambda, params.ro, params.beta, x);
% graderror = mean(abs(numgrad - grad));
% display(['SAE grad errors is ', num2str(graderror)]);

display('optimizing SAE...');

% SAE optimization:
load_weights_file = true;
if load_weights_file
   load('opttheta.mat');
   %opttheta = randTheta;
else
  [opttheta, cost, exitflag] = minFunc(@(theta) sparseAutoencoderCost(theta, params.visibleSize, params.hiddenSize, params.lambda, params.ro, params.beta, x), randTheta, options);
  save('opttheta.mat', 'opttheta');
end
%opttheta = rand(size(randTheta)) * (sqrt(6) / sqrt(size(randTheta, 1) + size(randTheta, 2)));

% reshape visualize weights
W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
%W2 = reshape(opttheta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = opttheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
%b2 = opttheta(2*hiddenSize*visibleSize+hiddenSize+1:end);
W = W1;
b = b1;
display_network(W');

%% ======================================================================

%% STEP 3: Extract Features from the Supervised Dataset
% pre-multiply the weights with whitening matrix, equivalent to whitening
% each image patch before applying convolution. V should be the same V
% returned by the zca2 when you whiten the patches.

% It seams that this line, as well as finding W using RICA do not
% have significant effect on accuracy. Using random W gives approximatelly
% same results. And using all 10 classes gives worst results then CNN. So
% either I'm doing something wrong, or all this does not make sense.

W = reshape(W, params.numFeatures, params.patchWidth, params.patchWidth);
W = permute(W, [2,3,1]);

%  setting up convolutional feed-forward. You do need to modify this code.
filterDim = params.patchWidth;
poolDim = 5;
numFilters = params.numFeatures;
trainImages=reshape(trainData, imgSize, imgSize, size(trainData, 2));
testImages=reshape(testData, imgSize, imgSize, size(testData, 2));
%  Compute convolutional responses
%  TODO: You will need to complete feedfowardRICA.m 
trainAct = feedfowardSAE(filterDim, poolDim, numFilters, trainImages, W, b);
testAct = feedfowardSAE(filterDim, poolDim, numFilters, testImages, W, b);
%  reshape the responses into feature vectors
featureSize = size(trainAct,1)*size(trainAct,2)*size(trainAct,3);
trainFeatures = reshape(trainAct, featureSize, size(trainData, 2));
testFeatures = reshape(testAct, featureSize, size(testData, 2));
%% ======================================================================
%% STEP 4: Train the softmax classifier

numClasses  = 5; % doing 5-class digit recognition
% initialize softmax weights randomly
randTheta2 = randn(numClasses - 1, featureSize)*0.01;  % 1/sqrt(params.n);
randTheta2 = randTheta2 ./ repmat(sqrt(sum(randTheta2.^2,2)), 1, size(randTheta2,2));
randTheta2 = randTheta2';
randTheta2 = randTheta2(:);

%  Use minFunc and softmax_regression_vec from the previous exercise to 
%  train a multi-class classifier. 
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 3000;

% optimize
%%% YOUR CODE HERE %%%

% display('checking SOFTMAX gradients');
% numgrad = computeNumericalGradient(@(theta) softmax_regression_vec(theta, trainFeatures, trainLabels), randTheta2);
% [~, grad] = softmax_regression_vec(randTheta2, trainFeatures, trainLabels);
% graderror = mean(abs(numgrad - grad));
% display(['SOFTMAX grad errors is ', num2str(graderror)]);

[opttheta, cost, exitflag] = minFunc(@(theta) softmax_regression_vec(theta, trainFeatures, trainLabels), randTheta2, options);
opttheta = reshape(opttheta, [featureSize, numClasses - 1]);
opttheta = [opttheta, zeros(featureSize, 1)];
%%======================================================================
%% STEP 5: Testing 
% Compute Predictions on tran and test sets using softmaxPredict
% and softmaxModel
%%% YOUR CODE HERE %%%
[~, train_pred] = max(opttheta' * trainFeatures, [], 1);
[~, pred] = max(opttheta' * testFeatures, [], 1);
% Classification Score
fprintf('Train Accuracy: %f%%\n', 100*mean(train_pred(:) == trainLabels(:)));
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));
% You should get 100% train accuracy and ~99% test accuracy. With random
% convolutional weights we get 97.5% test accuracy. Actual results may
% vary as a result of random initializations
