function [sValidaton]=runTS(path, dataName, delay, hiddenLayerSize)
% ***********************************************************************
% (c) 2014 Douglas Kruger
% ECE 626 - Advanced Neural Networks
% ***********************************************************************
myDebug=0;

if myDebug==1
    clear all; close all;    
    % Create a Nonlinear Autoregressive Network
    delay = 2; % 1-6 step 1
    hiddenLayerSize = 5; %10-40 - step 10
    dataName='Mackey-Glass'
    %dataName='SantaFeLaserA';
    path='C:\ECE626\';
end

% Configuration
folds = 5;           % Break sample into 5 folds
feedbackDelays = 1:delay;

% Read the original data file
origSeries = transpose(mapminmax(csvread(strcat(path, dataName,'.dat')),-1,1));

% Count the elements in the time series
nElements = size(origSeries,2);

% Partition the bottom 2/3 for training
iTrain=origSeries(1:nElements*2/3);

% Keep the top 1/3 for testing
iTest=origSeries(nElements*2/3+1:nElements);

% Count the elements in the time series
nElementsTrain = size(iTrain,2);
nElementsTest = size(iTest,2);
        
% narnet(feedbackDelays,hiddenSizes,trainFcn)
net = narnet(feedbackDelays,hiddenLayerSize);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;

% Prepare xvalidation
iTrainIndex = vec2ind(iTrain);
foldIndTrain = crossvalind('Kfold',iTrainIndex,folds);
 
% Keep the array to grab the errors
sValidaton=zeros(4,folds); 
 
% Loop through all the crossvalidation folds
for loop = 1:folds
    %Break Sample set into Training / Validation
    [iTraining, iValidation] = getSamplesTS(iTrain, folds, loop);

    % Prepare the Data for Training and Simulation
    % The function PREPARETS prepares timeseries data for a particular network,
    % shifting time by the minimum amount to fill input states and layer states.
    % Using PREPARETS allows you to keep your original time series data unchanged, while
    % easily customizing it for networks with differing numbers of delays, with
    % open loop or closed loop feedback modes.
    [inputs,inputStates,layerStates,targets] = preparets(net,{},{},num2cell(iTraining));

    % Train the Network
    [net,tr] = train(net,inputs,targets,inputStates,layerStates);
%    outputs = net(inputs,inputStates,layerStates);
 %   errors = gsubtract(targets,outputs);
  %  figure, plotresponse(targets,outputs)
    
    % Validate
    [inputs,inputStates,layerStates,targets] = preparets(net,{},{},num2cell(iValidation));
    outputs = net(inputs,inputStates,layerStates);
 %   errors = gsubtract(targets,outputs);
%    figure, plotresponse(targets,outputs);
    sValidaton(:,loop)=transpose(calcStats(cell2mat(targets),cell2mat(outputs)));
end

paramString=strcat('-d',num2str(delay),'-n',num2str(hiddenLayerSize));

% Test the Network
[inputs,inputStates,layerStates,targets] = preparets(net,{},{},num2cell(iTest));
outputs = net(inputs,inputStates,layerStates);
errors = gsubtract(targets,outputs);

x=linspace(1,size(targets,2),size(targets,2))
figure,plot(x,cell2mat(targets),'-k',x,cell2mat(outputs),'-r',x,cell2mat(errors),'-m');
title(strcat(dataName,' Actual and Predictions',paramString));
xlabel('Data Points');
ylabel('Value');
legend('Location','EastOutside','Actual','Predicted','Error');
saveas(gcf,strcat(path,dataName,'-TS-Actual',paramString,'.png'));

figure, h=plotresponse(targets,outputs);
saveas(h,strcat(path,dataName,'-TS-Response',paramString,'.png'));
%performance = perform(net,targets,outputs)

% Plots - Regression
figure, h=plotregression(targets,outputs);
saveas(h,strcat(path,dataName,'-TS-Regression',paramString,'.png'));

% Plots - Error Histogram
figure, h=ploterrhist(errors);
saveas(h,strcat(path,dataName,'-TS-ErrHistogram',paramString,'.png'));

% Plots - Error Correlation
figure, h=ploterrcorr(errors);
saveas(h,strcat(path,dataName,'-TS-ErrCorr',paramString,'.png'));

% Plots - Error Auto Correlation
figure, h=plotinerrcorr(inputs,errors)
saveas(h,strcat(path,dataName,'-TS-InErrCorr',paramString,'.png'));

% Plots - Training Errors
figure
x=linspace(1,folds,folds)
subplot(2,2,1)
plot(x,sValidaton(3,:),'-k');
xlabel('Data Points');
ylabel('NDEI');
subplot(2,2,2)
plot(x,sValidaton(4,:),'-k');
xlabel('Data Points');
ylabel('AE');
subplot(2,2,3)
plot(x,sValidaton(1,:),'-k');
xlabel('Data Points');
ylabel('MSE');
subplot(2,2,4)
plot(x,sValidaton(2,:),'-k');
xlabel('Data Points');
ylabel('NMSE');
suptitle(strcat(dataName,' Training Errors',paramString));
saveas(gcf,strcat(path,dataName,'-TS-TrainingErrors',paramString,'.png'));

% Exit the open plots
close all;