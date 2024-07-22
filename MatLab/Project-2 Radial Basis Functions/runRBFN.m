function [TPRAverage, FPRAverage]=runRBFN(dataset, spreadConstant, maxNeurons)
% ***********************************************************************
% (c) 2014 Douglas Kruger
% ECE 626 - Advanced Neural Networks
% ***********************************************************************
% Clear the past data
%close all; clear all; clc; format compact;
 
% Configuration
folds = 10;          % Break sample into 10 folds
errorGoal = 0;       % sum-squared error goal => Set to zero to allow reaching of max neurons

if dataset == 1
    % Wine Dataset
    % Normalize the data
    load wine_dataset
    iMaster = mapminmax(wineInputs,0,1);
    tMaster = mapminmax(wineTargets,0,1);
elseif dataset == 2
    % Wisconsin Breast Cancer Dataset
    cancerInputs=csvread('C:\ECE626\RBFN Example\breast-cancer-wisconsin.csv')
    cancerTargets=csvread('C:\ECE626\RBFN Example\breast-cancer-wisconsin-class.csv')
    % Normalize the data
    iMaster = mapminmax(cancerInputs,0,1);
    tMaster = mapminmax(cancerTargets,0,1);
else
    r=4;
    w=2;
    d=-1;
    N=300;
    seed=2;
    display=1;
    [moonInput moonTarget]=generate_two_moons(r,w,d,N,seed,display)
    % Rotate Array 90 degrees
    for loopx = 1:size(moonInput,1)
        for loopy = 1:size(moonInput,2)
            mInput(loopy,loopx)=moonInput(loopx,loopy);
        end
        if moonTarget(loopx) == 1
            mTarget(1, loopx) = 1;
            mTarget(2, loopx) = 0;
        else
            mTarget(2, loopx) = 0;
            mTarget(2, loopx) = 1;
        end
    end
    iMaster = mapminmax(mInput,0,1);
    tMaster = mapminmax(mTarget,0,1);
end

masterIndex = vec2ind(tMaster);
[classes, instances] = size(tMaster);
 
% Generate a list for 10-Fold Cross Validation to pull out the test data
% Break data into 10 sets of size n/10.
% Train on 9 datasets and test on 1.
% Repeat 10 times and take a mean accuracy.
foldIndMaster = crossvalind('Kfold',masterIndex,folds);
 
% Initialize True Positive, True Negative, False Positive, False Negative variables
TP(folds,classes) = 0;
TN(folds,classes) = 0;
FP(folds,classes) = 0;
FN(folds,classes) = 0; 
TC(folds)=0;
TF(folds)=0;

% Loop through all the crossvalidation folds
for loop = 1:folds
    %Break Sample set into Training / Validation
    [iTraining, tTraining, iValidation, tValidation] = getSamples(iMaster, tMaster, foldIndMaster, loop);
 
    % Define the nework and train it with the training data from the
    % crossvalidation
    net=newrb(iTraining,tTraining,errorGoal,spreadConstant,maxNeurons);
   
    % Test the Network with the training data (remaining section of the
    % cross validation)
    oValidation = net(iValidation);
  
    % determine the class (use the index level) for the output and target
    % basically select the highest value for the output
    oValidationClass = vec2ind(oValidation);
    tValidationClass = vec2ind(tValidation);
 
    % For all of the samples (instances)
    % Determine the True Positive, True Negative, False Positive, False
    % Negative values for each class
    %[features, instances] = size(iValidation,2);
    instances = size(iValidation,2);
    for class = 1:classes
       for innerLoop = 1:instances
            if (oValidationClass(innerLoop) == class) 
                if (tValidationClass(innerLoop) == class)
                    TP(loop, class) = TP(loop,class) + 1;
                elseif (tValidationClass(innerLoop) ~= class)
                    FN(loop,class) = FN(loop,class) + 1;
                end
            elseif (oValidationClass(innerLoop) ~= class)
                if (tValidationClass(innerLoop) ~= class)
                    TN(loop,class) = TN(loop,class) + 1;
                elseif (tValidationClass(innerLoop) == class)
                    FP(loop,class) = FP(loop,class) + 1;
                end
            end
       end
       % Evaluate all classes for each loop
       % need to do this outside of the previous loop
       TPR(loop,class) = TP(loop,class) / (TP(loop,class) + FN(loop,class));
       FPR(loop,class) = FP(loop,class) / (FP(loop,class) + TN(loop,class));
    end
end

% Calculate the mean and standard deviations for all the Accuracy, TPR, FPR
% for each 10 fold experiment
for class = 1:classes
    TPRAverage(class) = mean(TPR(:,class));
    FPRAverage(class) = mean(FPR(:,class));
end