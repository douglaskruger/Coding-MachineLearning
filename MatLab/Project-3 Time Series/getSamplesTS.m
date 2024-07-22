function [iSamples, iValidation] = getSamplesTS(iMaster, folds, foldindex)
% ***********************************************************************
% (c) 2014 Douglas Kruger
% ECE 626 - Advanced Neural Networks
% ***********************************************************************

% Copy the inputs from the sample set
iSamples = iMaster;
iValidation = iMaster;

% Dynamically get the number of features and instances
instances = size(iMaster,2);

% remove the 'test' sample as per the 10 fold cross validation
% iteration from the last record down - required as we delete
% members in the array
for innerLoop = instances:-1:1
    if innerLoop >= (instances/folds*(foldindex-1)) && (innerLoop <= (instances/folds*foldindex))
        iSamples(:,innerLoop) = [];
    else
        iValidation(:,innerLoop) = [];
    end
end 