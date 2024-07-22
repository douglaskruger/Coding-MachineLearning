function [iSamples, tSamples, iValidation, tValidation] = getSamples(iMaster, tMaster, foldIndexMaster, index)
% ***********************************************************************
% (c) 2014 Douglas Kruger
% ECE 626 - Advanced Neural Networks
% ***********************************************************************
    % Copy the inputs from the sample set
    iSamples = iMaster;
    tSamples = tMaster;
    iValidation = iMaster;
    tValidation = tMaster;
    
    % Dynamically get the number of features and instances
    instances = size(iMaster,2);
 
    % remove the 'test' sample as per the 10 fold cross validation
    % iteration from the last record down - required as we delete
    % members in the array
    for innerLoop = instances:-1:1
        if foldIndexMaster(innerLoop) == index
            iSamples(:,innerLoop) = [];
            tSamples(:,innerLoop) = [];
        else
            iValidation(:,innerLoop) = [];
            tValidation(:,innerLoop) = [];
        end
    end