% ***************************************************************************************
% ECE 623 - Data Exploration and Evolutionary Computing
% University of Alberta
% (c) 2014 Douglas Kruger
% ***************************************************************************************
clc; close all; clear all

% Constant - Parameters
path='C:\ECE623\project\';
iFile='train';
oFile='features';
%spread = [0.1 0.5 0.75 1 1.25 1.5 2 2.5 5 10 20 50];
%neuron = [10 20 50 200 300]; 
sampleSize=8000;
spread = [1 2 3 4 5 6 7 8 9 10 11];
neuron = [30 40 50 70 80 100 120]; 

% Extract the features based on the input data
[classBinary, features]=extractFeatures(path,iFile,oFile);
%generateImages('C:\ECE623\project\','train',0);

% Normalize the data
%iMaster = mapminmax(features',0,1);
%tMaster = mapminmax(classBinary',0,1);

iMaster = mapminmax(features(1:sampleSize,:)',0,1);
tMaster = mapminmax(classBinary(1:sampleSize,:)',0,1);
clear features classBinary;
counter=1;
for spreadLoop = 1:size(spread,2)
    for neuronLoop = 1:size(neuron,2)
        fprintf('Processing - spreadLoop %i(%i), neuronLoop %i(%i)',...
            spreadLoop,spread(spreadLoop),neuronLoop,neuron(neuronLoop));
        [performance, TPR, FPR]=runRBFN(iMaster, tMaster, spread(spreadLoop),neuron(neuronLoop));
        for class = 1:size(TPR,2)
            wTPR(counter,class) = TPR(class);
            wFPR(counter,class) = FPR(class);
        end
        wPRindex(counter,1) = neuronLoop;
        wPRindex(counter,2) = spreadLoop;
        factor(counter,1)=spread(spreadLoop);
        factor(counter,2)=neuron(neuronLoop);
        factor(counter,3)=performance;
        counter = counter+1;
    end
end

%outputFile=strcat(path,'performance.csv');
%csvwrite(outputFile,[factor performance]'); % Write the data
outputFile=strcat(path,'TPR.csv');
csvwrite(outputFile,[factor wTPR]'); % Write the data
outputFile=strcat(path,'FPR.csv');
csvwrite(outputFile,[factor wFPR]'); % Write the data 