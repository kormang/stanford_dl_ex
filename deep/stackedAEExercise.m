%% CS294A/CS294W Stacked Autoencoder Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  sstacked autoencoder exercise. You will need to complete code in
%  stackedAECost.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises. You will need the initializeParameters.m
%  loadMNISTImages.m, and loadMNISTLabels.m files from previous exercises.
%  
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file. 
%
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.
addpath(genpath('..'))

inputSize = 28 * 28;
numClasses = 10;
hiddenSizeL1 = 196;    % Layer 1 Hidden Size
hiddenSizeL2 = 196;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Load data from the MNIST database
%
%  This loads our training data from the MNIST database files.

% Load MNIST database files
mnistData   = loadMNISTImages('../common/train-images-idx3-ubyte');
mnistLabels = loadMNISTLabels('../common/train-labels-idx1-ubyte');

unsupervisedTrainData = mnistData(:, 1:50000);

trainData = mnistData(:, 1:50000);
trainLabels = mnistLabels(1:50000);

trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.

%  Randomly initialize the parameters
sae1Theta = initializeSAEParameters(hiddenSizeL1, inputSize);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                You should store the optimal parameters in sae1OptTheta
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 300;

[sae1OptTheta, cost, exitflag] = minFunc(@(theta) sparseAutoencoderCost(theta, inputSize, hiddenSizeL1, lambda, sparsityParam, beta, unsupervisedTrainData), sae1Theta, options);
%sae1OptTheta = rand(size(sae1Theta));

% -------------------------------------------------------------------------



%%======================================================================
%% STEP 2: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);

%  Randomly initialize the parameters
sae2Theta = initializeSAEParameters(hiddenSizeL2, hiddenSizeL1);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the second layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL2" and an inputsize of
%                "hiddenSizeL1"
%
%                You should store the optimal parameters in sae2OptTheta

[sae2OptTheta, cost, exitflag] = minFunc(@(theta) sparseAutoencoderCost(theta, hiddenSizeL1, hiddenSizeL2, lambda, sparsityParam, beta, sae1Features), sae2Theta, options);
%sae2OptTheta = rand(size(sae2Theta));


% -------------------------------------------------------------------------


%%======================================================================
%% STEP 3: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.

[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * (numClasses - 1), 1);


%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);

[saeSoftmaxOptTheta, cost, exitflag] = minFunc(@(theta) softmax_regression_vec(theta, sae2Features, trainLabels'), saeSoftmaxTheta, options);
saeSoftmaxOptTheta = reshape(saeSoftmaxOptTheta, hiddenSizeL2, numClasses - 1);
saeSoftmaxOptTheta = [saeSoftmaxOptTheta, zeros(size(saeSoftmaxOptTheta, 1), 1)];
figure
display_network(saeSoftmaxOptTheta);
print -dpng saeSoftmaxOptTheta.png
saeSoftmaxOptTheta = saeSoftmaxOptTheta';
saeSoftmaxOptTheta = saeSoftmaxOptTheta(:);
% -------------------------------------------------------------------------



%%======================================================================
%% STEP 5: Finetune softmax model

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.W = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.W = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);


display_network(stack{1}.W');
print -dpng pre1s1W.png
display_network(stack{2}.W');
print -dpng pre1s2W.png

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
%
%

[stackedAEOptTheta, cost, exitflag] = minFunc(@(theta) stackedAECost(theta, inputSize, hiddenSizeL2, numClasses, netconfig, lambda, trainData, trainLabels), ...
                                                stackedAETheta, options);


dispOptSoftmaxTheta = reshape(stackedAEOptTheta(1:hiddenSizeL2*numClasses), numClasses, hiddenSizeL2);
stack = params2stack(stackedAEOptTheta(hiddenSizeL2*numClasses+1:end), netconfig);

display_network(dispOptSoftmaxTheta');
print -dpng dispOptSoftmaxTheta.png

display_network(stack{1}.W');
print -dpng pre2s1W.png
display_network(stack{2}.W');
print -dpng pre2s2W.png
% -------------------------------------------------------------------------



%%======================================================================
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
testData = mnistData(:, 50001:end);
testLabels = mnistLabels(50001:end);

testLabels(testLabels == 0) = 10; % Remap 0 to 10


[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
