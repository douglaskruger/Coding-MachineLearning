%% ***************************************************************************************
% ECE 623 - Data Exploration and Evolutionary Computing
% University of Alberta
% (c) 2014 Douglas Kruger
% ***************************************************************************************

% Run Locally
close all; clear all; clc; format compact;
[classBinary, features]=extractFeatures('C:\ECE623\project\','train','features');
%generateImages('C:\ECE623\project\','train',0);
spreadConstant =2.5;
maxNeurons = 200;
iTraining = mapminmax(features(1:8000,:)',0,1);
tTraining = mapminmax(classBinary(1:8000,:)',0,1);
iValidation = mapminmax(features(8001:42000,:)',0,1);
tValidation = mapminmax(classBinary(8001:42000,:)',0,1);
clear features classBinary;
[classes, instances] = size(tTraining);

% Configuration
errorGoal = 0;       % sum-squared error goal => Set to zero to allow reaching of max neurons

% Initialize True Positive, True Negative, False Positive, False Negative variables
TP(classes) = zeros;
TN(classes) = zeros;
FP(classes) = zeros;
FN(classes) = zeros;

% Define the nework and train it with the training data
net=newrb(iTraining,tTraining,errorGoal,spreadConstant,maxNeurons,maxNeurons);
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', 'plotregression', 'plotconfusion'};

% Test the Network with the training data (remaining section of the cross validation)
oValidation = net(iValidation);

% Test the Network
errors = gsubtract(tValidation,oValidation);
performance = perform(net,tValidation,oValidation);

% View the Network
view(net)

% Plots
figure, plotconfusion(tValidation,oValidation)
figure, ploterrhist(errors)

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
                TP(class) = TP(class) + 1;
            elseif (tValidationClass(innerLoop) ~= class)
                FN(class) = FN(class) + 1;
            end
        elseif (oValidationClass(innerLoop) ~= class)
            if (tValidationClass(innerLoop) ~= class)
                TN(class) = TN(class) + 1;
            elseif (tValidationClass(innerLoop) == class)
                FP(class) = FP(class) + 1;
            end
        end
    end
    % Evaluate all classes for each loop
    % need to do this outside of the previous loop
    TPR(class) = TP(class) / (TP(class) + FN(class));
    FPR(class) = FP(class) / (FP(class) + TN(class));
end

% Calculate the mean and standard deviations for all the Accuracy, TPR, FPR
% for each X fold experiment
for class = 1:classes
    TPRAverage(class) = mean(TPR(:,class));
    FPRAverage(class) = mean(FPR(:,class));
end
performanceAverage = mean(performance);

