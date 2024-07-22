% ***********************************************************************
% (c) Douglas Kruger
% ECE 626 - Advanced Neural Networks
% ***********************************************************************
% Clear the past data
close all;
clear all;
clc;
format compact;
 
% Load the normalized and encoded input and target data
load wine_dataset
 
% Normalize the inputs
masterInputs = mapminmax(wineInputs,0,1);
masterTargets = mapminmax(wineTargets,0,1);
 
% Dynamically get the number of features and instances
[~, instances] = size(masterInputs);
 
% Remove 10% of the original sample - keep to the side as the final test
% case
 
% Label the three choices for our data set
wineryData = {'Winery1'; 'Winery2'; 'Winery3'};
 
% Generate an array of characters for the target output
% This is used by the cross validation
masterWineryLabel=cell(1);
for sample = 1:instances
    for class = 1:3
        if masterTargets(class,sample) == 1
            masterWineryLabel(sample) = wineryData(class);
        end
    end
end
 
% Generate a list for 10-Fold Cross Validation to pull out the test data
% Break data into 10 sets of size n/10.
% Train on 9 datasets and test on 1.
% Repeat 10 times and take a mean accuracy.
folds = 10;
masterTenFoldIndices = crossvalind('Kfold',masterWineryLabel,folds);
 
% Copy the inputs from the master
sampleInputs = masterInputs;
sampleTargets = masterTargets;
testInputs = masterInputs;
testTargets = masterTargets;
    
% keep index 1 as primary test data, merge the rest
for innerLoop = instances:-1:1
    if masterTenFoldIndices(innerLoop) == 1
        sampleInputs(:,innerLoop) = [];
        sampleTargets(:,innerLoop) = [];
    else
        testInputs(:,innerLoop) = [];
        testTargets(:,innerLoop) = [];
    end
end
 
% Dynamically get the number of features and instances
[features, instances] = size(sampleInputs);
 
% Generate an array of characters for the target output
% This is used by the cross validation
sampleWineryLabel=cell(1);
for sample = 1:instances
    for class = 1:3
        if sampleTargets(class,sample) == 1
            sampleWineryLabel(sample) = wineryData(class);
        end
    end
end
% Generate a list for 10-Fold Cross Validation for the ten samples
% Break data into 10 sets of size n/10.
% Train on 9 datasets and test on 1.
% Repeat 10 times and take a mean accuracy.
sampleTenFoldIndices = crossvalind('Kfold',sampleWineryLabel,folds);
 
% Initialize variables
TP(10,3) = 0;
TN(10,3) = 0;
FP(10,3) = 0;
FN(10,3) = 0;
TC(10)=0;
for loop = 1:10
    % Copy the inputs from the sample set
    inputs = sampleInputs;
    targets = sampleTargets;
    inputsValidation = sampleInputs;
    targetsValidation = sampleTargets;
    
    % Dynamically get the number of features and instances
    [~, instances] = size(inputs);
 
    % remove the 'test' sample as per the 10 fold cross validation
    for innerLoop = instances:-1:1
        if sampleTenFoldIndices(innerLoop) == loop
            inputs(:,innerLoop) = [];
            targets(:,innerLoop) = [];
        else
            inputsValidation(:,innerLoop) = [];
            targetsValidation(:,innerLoop) = [];
        end
    end
 
    % Create a Pattern Recognition Network
    hiddenLayerSize = [5];
    
    % net = patternnet(hiddenLayerSize,'traingdm');  
    net = patternnet(hiddenLayerSize,'traingdm');  
  
    % Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    % Default:  {'removeconstantrows', mapminmax}
    net.inputs{1}.processFcns = {'removeconstantrows'};
    net.outputs{2}.processFcns = {'removeconstantrows'};
 
    % Setup Division of Data for Training, Validation, Testing
    % For a list of all data division functions type: help nndivide
    % net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideFcn = 'dividetrain';  % Divide data randomly
    net.divideMode = 'sample';  % Divide up every sample
   
    %   Default Function Parameters for 'traingdm'
    %Show Training Window Feedback   showWindow: true
    %Show Command Line Feedback showCommandLine: false
    %Command Line Frequency                show: 25
    %Maximum Epochs                      epochs: 1000
    %Maximum Training Time                 time: Inf
    %Performance Goal                      goal: 0
    %Minimum Gradient                  min_grad: 1e-05
    %Maximum Validation Checks         max_fail: 6
    %Learning Rate                           lr: 0.01
    %Momentum Constant                       mc: 0.9
    
    % Cap the epochs to 50000
    net.trainParam.epochs = 50000; 
    net.trainParam.lr = 0.1;
    net.trainParam.goal = 0.0015;

    % Choose a Performance Function
    % For a list of all performance functions type: help nnperformance
    net.performFcn = 'mse';  % Mean squared error
  
    % Train the Network
    [net,tr] = train(net,inputs,targets);
 
    % Test the Network
    outputsValidation = net(inputsValidation);
    errors = gsubtract(targetsValidation,outputsValidation);
    performance = perform(net,targetsValidation,outputsValidation);
  
    % determine the class (use the index level) for the output and target
    outputsValidationClass = vec2ind(outputsValidation);
    targetsValidationClass = vec2ind(targetsValidation);
 
    [features, instances] = size(inputsValidation);
    for n = 1:instances
        for class = 1:3
            if (outputsValidationClass(n) == class) 
                if (targetsValidationClass(n) == class)
                    TP(loop, class) = TP(loop,class) + 1;
                elseif (targetsValidationClass(n) ~= class)
                    FN(loop,class) = FN(loop,class) + 1;
                end
            elseif (outputsValidationClass(n) ~= class)
                if (targetsValidationClass(n) ~= class)
                    TN(loop,class) = TN(loop,class) + 1;
                elseif (targetsValidationClass(n) == class)
                    FP(loop,class) = FP(loop,class) + 1;
                end
            end
        end
    end
    
    % Evaluate all classes - need to do this outside of the previous loop
    for m = 1:3
        TPR(loop,m) = TP(loop,m) / (TP(loop,m) + FN(loop,m));
        FPR(loop,m) = FP(loop,m) / (FP(loop,m) + TN(loop,m));
    end
    
    % Accuracy = Total correct / Total instances
    for n = 1:instances
        if (outputsValidationClass(n) == targetsValidationClass(n))
            TC(loop) = TC(loop) + 1;
        end
    end
    accuracy(loop) = TC(loop) / instances;    
 
end
% Calculate the mean and standard deviations for all the Accuracy, TPR, FPR
% for each 10 fold experiment
A_accuracyAverage = mean(accuracy);
A_accuracyStd = std(accuracy);
for m = 1:3
    A_TPRAverage(m) = mean(TPR(:,m));
    A_TPRStdev(m) = std(TPR(:,m));
    A_FPRAverage(m) = mean(FPR(:,m));
    A_FPRStdev(m) = std(FPR(:,m));
end
 
% *************************************************
% Check the out of sample data
% *************************************************
% Train the Network based on the parameters above
sampleInputs = masterInputs;
sampleTargets = masterTargets;
testInputs = masterInputs;
testTargets = masterTargets;
[net,tr] = train(net,sampleInputs,sampleTargets);
 
% Test the Network and evaluate using the MSE
testOutputs = net(testInputs);
errors = gsubtract(testTargets,testOutputs);
performanceOutofSample = perform(net,testTargets,testOutputs);
