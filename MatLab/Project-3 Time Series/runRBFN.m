function [sValidaton]=runRBFN(path,dataName, delay, m, spreadConstant, maxNeurons)
% ***********************************************************************
% (c) 2014 Douglas Kruger
% ECE 626 - Advanced Neural Networks
% ***********************************************************************
myDebug=0
if myDebug==1
    clear all; close all;
    delay=3
    m=8
    spreadConstant=1
    maxNeurons=20
    dataName='Mackey-Glass'
    %dataName='SantaFeLaserA';
    path='C:\ECE626\';
end

% Configuration
folds = 5;           % Break sample into 5 folds
errorGoal = 0;       % sum-squared error goal => Set to zero to allow reaching of max neurons

% Read the original data file
% Normalize and Rotate the matrix
origInput = transpose(mapminmax(csvread(strcat(path,dataName,'.dat')),-1,1));
    
% Count the elements in the time series
nElements = size(origInput,2);

% Partition the bottom 2/3 for training
iTrain=origInput(1:nElements*2/3);

% Keep the top 1/3 for testing
iTest=origInput(nElements*2/3+1:nElements);

% Count the elements in the time series
nElementsTrain = size(iTrain,2);
nElementsTest = size(iTest,2);

% Create delay vector (elements - d(m-1)) for Train
nArrayTraining = nElementsTrain - delay*(m-1);
for loopDV = 1:1:nArrayTraining
    for loopElementDV = 1:1:m-1
        myIndex=loopDV+(m-1)*delay-loopElementDV*delay;
        iTrainDV(loopElementDV,loopDV)=iTrain(myIndex);
    end
    tTrainDV(loopDV)=iTrain(loopDV+(m-1)*delay);
end
% Create delay vector (elements - d(m-1)) for Test
nArrayTest = nElementsTest - delay*(m-1);
for loopDV = 1:1:nArrayTest
    for loopElementDV = 1:1:m-1
        myIndex=loopDV+(m-1)*delay-loopElementDV*delay;
        iTestDV(loopElementDV,loopDV)=iTest(myIndex);
    end
    tTestDV(loopDV)=iTest(loopDV+(m-1)*delay);
end

% Generate a list for 10-Fold Cross Validation to pull out the test data
% Break data into 10 sets of size n/10.
% Train on 9 datasets and test on 1.
% Repeat 10 times and take a mean accuracy.
trainDVIndex = vec2ind(tTrainDV);
foldIndTrain = crossvalind('Kfold',trainDVIndex,folds);

% Keep the array to grab the errors
sValidaton=zeros(4,folds); 

% Loop through all the crossvalidation folds
for loop = 1:folds
    %Break Sample set into Training / Validation
    [iTraining, tTraining, iValidation, tValidation] = getSamples(iTrainDV,tTrainDV, foldIndTrain, loop);

    % Define the nework and train it with the training data from the
    % crossvalidation
    net=newrb(iTraining, tTraining,errorGoal,spreadConstant,maxNeurons);
    net.plotFcns = {'plotperform','plottrainstate','plotresponse', ...
      'ploterrcorr', 'plotinerrcorr'};
    % Test the Network with the training data (remaining section of the
    % cross validation)
    oValidation = net(iValidation); 
    pValidation = perform(net,tValidation,oValidation)
    sValidaton(:,loop)=transpose(calcStats(tValidation,oValidation));
end

% Retrain the network and determine the errors
%net=newrb(iTrainDV, tTrainDV,errorGoal,spreadConstant,maxNeurons);
oTestDV=net(iTestDV);
eTestDV = gsubtract(tTestDV,oTestDV);
pTestDV = perform(net,tTestDV,oTestDV);
sTestDV(:,folds+1)=transpose(calcStats(tTestDV,oTestDV));

% View the Network
%view(net)
paramString=strcat('-m',num2str(m),'-d',num2str(delay),'-s',num2str(spreadConstant),'-n',num2str(maxNeurons));

% Plots - Regression
figure, h=plotregression(tTestDV,oTestDV);
saveas(h,strcat(path,dataName,'-RBF-Regression',paramString,'.png'));

% Plots - Error Histogram
figure, h=ploterrhist(eTestDV);
saveas(h,strcat(path,dataName,'-RBF-ErrHistogram',paramString,'.png'));

% Plots - Error Correlation
figure, h=ploterrcorr(eTestDV);
saveas(h,strcat(path,dataName,'-RBF-ErrCorr',paramString,'.png'));

% Plots - Error Auto Correlation
figure, h=plotinerrcorr(iTestDV,eTestDV)
saveas(h,strcat(path,dataName,'-RBF-InErrCorr',paramString,'.png'));

% Plots - Actual, Predictions, Errors
x=linspace(1,nArrayTest,nArrayTest)
figure,plot(x,tTestDV,'-k',x,oTestDV,'-r',x,eTestDV,'-m');
title(strcat(dataName,' Actual and Predictions',paramString));
xlabel('Data Points');
ylabel('Value');
legend('Location','EastOutside','Actual','Predicted','Error');
saveas(gcf,strcat(path,dataName,'-RBF-Actual',paramString,'.png'));

figure, h=plotresponse(num2cell(tTestDV),num2cell(oTestDV));
saveas(h,strcat(path,dataName,'-RBF-Response',paramString,'.png'));

% Plots - Training Errors
figure
x=linspace(1,folds,folds)
subplot(2,2,1)
plot(x,sValidaton(3,:),'-k');
xlabel('Data Points');
ylabel('NDEI');
suptitle(strcat(dataName,' Training Errors',paramString));
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
saveas(gcf,strcat(path,dataName,'-RBF-Training',paramString,'.png'));

% Plots - Delay Space
x1=iTrain(1:nElementsTrain-delay)
x2=iTrain(delay:nElementsTrain-1)
figure,plot(x1,x2,'-k');
title(strcat(dataName,' State Space-Delay=',num2str(delay)));
xlabel('x(t)');
ylabel(strcat('x(t-',num2str(delay),')'));
saveas(gcf,strcat(path,dataName,'-RBF-Statespace-Delay=',num2str(delay),'.png'));

% Save variables from instance for later
save(strcat(path,dataName,'-RBF-variables',paramString,'.mat'));

% Exit the open plots
close all;