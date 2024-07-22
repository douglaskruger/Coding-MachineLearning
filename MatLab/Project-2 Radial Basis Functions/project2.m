% ***********************************************************************
% (c) 2014 Douglas Kruger
% ECE 626 - Advanced Neural Networks
% ***********************************************************************

% Loop through all the crossvalidation folds
close all; clear all; clc; format compact;

counter=1;
dataset=1;
spread = [0.25 0.5 1 2 3 5 10 20];
neuron = [5 10 20 50 100 150 200 300]; 
for spreadLoop = 1:size(spread,2)
    for neuronLoop = 1:size(neuron,2)
        fprintf('Processing - dataset %i, spreadLoop %i, neuronLoop %i',...
            dataset,spreadLoop,neuronLoop);
        [lTPR, lFPR]=runRBFN(dataset,spread(spreadLoop),neuron(neuronLoop));
        for class = 1:size(lTPR,2)
            wTPR(counter,class) = lTPR(class);
            wFPR(counter,class) = lFPR(class);
        end
        wPRindex(counter,1) = neuronLoop;
        wPRindex(counter,2) = spreadLoop;
        counter = counter+1;
    end
end
counter=1;
dataset=2;
for spreadLoop = 1:size(spread,2)
    for neuronLoop = 1:size(neuron,2)
        fprintf('Processing - dataset %i, spreadLoop %i, neuronLoop %i',...
            dataset,spreadLoop,neuronLoop);
        [lTPR, lFPR]=runRBFN(dataset,spread(spreadLoop),neuron(neuronLoop));
        for class = 1:size(lTPR,2)
            cTPR(counter,class) = lTPR(class);
            cFPR(counter,class) = lFPR(class);
        end
        cPRindex(counter,1) = neuronLoop;
        cPRindex(counter,2) = spreadLoop;
        counter = counter+1;
    end
end
counter=1;
dataset=3;
for spreadLoop = 1:size(spread,2)
    for neuronLoop = 1:size(neuron,2)
        fprintf('Processing - dataset %i, spreadLoop %i, neuronLoop %i',...
            dataset,spreadLoop,neuronLoop);
        [lTPR, lFPR]=runRBFN(dataset,spread(spreadLoop),neuron(neuronLoop));
        for class = 1:size(lTPR,2)
            mTPR(counter,class) = lTPR(class);
            mFPR(counter,class) = lFPR(class);
        end
        mPRindex(counter,1) = neuronLoop;
        mPRindex(counter,2) = spreadLoop;
        counter = counter+1;
    end
end