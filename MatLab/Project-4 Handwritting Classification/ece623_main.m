%% ***************************************************************************************
% ECE 623 - Data Exploration and Evolutionary Computing
% University of Alberta
% (c) 2014 Douglas Kruger
% ***************************************************************************************
%clear all; close all;
%[charClass, classBinary, features]=extractFeatures('C:\ECE623\project\','train','features');
%generateImages('C:\ECE623\project\','train',0);

inputs = features';
targets = classBinary';

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);

net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};
net.trainParam.epochs = 50000; 
net.trainParam.lr = 0.15;
net.trainParam.goal = 0.0001;

net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample

%Train on 70% of data = 14000
%Validate network performance with 5% of data = 1000
%Test network on 25% of data = 5000
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 5/100;
net.divideParam.testRatio = 25/100;

%net.trainFcn = 'trainscg';  % Scalable Conjugate Gradient
net.trainFcn = 'traingdm';  % Scalable Conjugate Gradient
net.performFcn = 'mse';  % Mean squared error
% Create buttons that plot performance, training state, error histogram,
% regression, and confusion
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', 'plotregression', 'plotconfusion'};

% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);

% Recalculate Training, Validation and Test Performance
trainTargets = targets .* tr.trainMask{1};
valTargets = targets .* tr.valMask{1};
testTargets = targets .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,outputs)
valPerformance = perform(net,valTargets,outputs)
testPerformance = perform(net,testTargets,outputs)

% View the Network
view(net)

% Plots
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, plotconfusion(targets,outputs)
%figure, ploterrhist(errors)

