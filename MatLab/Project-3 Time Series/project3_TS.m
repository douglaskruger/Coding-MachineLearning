% ***********************************************************************
% (c) 2014 Douglas Kruger
% ECE 626 - Advanced Neural Networks
% ***********************************************************************

% Loop through all the crossvalidation folds
close all; clear all; clc; format compact;

% Configuration
%path='C:\ECE626\';
path='C:\Users\kruger.WILLOWGLEN\Google Drive\MBA\ECE 626 - Advanced Neural\Project3aData\';
spread = [0.3 1 3];
neuron = [10 20 30];

for datasetLoop=1:1
    if datasetLoop==1
        % Mackey-Glass (m=14,d=10)
        delay=[6 10 14];
        m=[9 12 15];
        dataName='Mackey-Glass'
    else
        % SantaFe LaserA (m=8,d=3)
        delay=[2 3 5 10];
        m=[4 6 8 10];
        dataName='SantaFeLaserA';
    end
    for neuronLoop = 1:size(neuron,2)
        for delayLoop = 1:size(delay,2)
            fprintf('TS Processing - dataset=%i, delay=%i, neuron=%i',...
                datasetLoop, delay(delayLoop), neuron(neuronLoop));    
                Errors=runTS(path, dataName, delay(delayLoop), neuron(neuronLoop));
                TS_MSE_Mean(delayLoop,neuronLoop)=mean(Errors(1,:));
                TS_MSE_Std(delayLoop,neuronLoop)=std(Errors(1,:));
                TS_NMSE_Mean(delayLoop,neuronLoop)=mean(Errors(2,:));
                TS_NMSE_Std(delayLoop,neuronLoop)=std(Errors(2,:));
                TS_NDEI_Mean(delayLoop,neuronLoop)=mean(Errors(3,:));
                TS_NDEI_Std(delayLoop,neuronLoop)=std(Errors(3,:));
                TS_AE_Mean(delayLoop,neuronLoop)=mean(Errors(4,:));
                TS_AE_Std(delayLoop,neuronLoop)=std(Errors(4,:));
        end 
    end
    % Plots - Actual, Predictions, Errors
    x=linspace(1,delay(delayLoop),delay(delayLoop))
    figure
    subplot(2,4,1)
    plot(delay,TS_MSE_Mean(:,1),'-k',delay,TS_MSE_Mean(:,2),'-r',delay,TS_MSE_Mean(:,3),'-m');
    xlabel('Data Points');
    ylabel('MSE');
    subplot(2,4,2)
    plot(delay,TS_MSE_Std(:,1),'-k',delay,TS_MSE_Std(:,2),'-r',delay,TS_MSE_Std(:,3),'-m');
    xlabel('Data Points');
    ylabel('MSE Std');
    subplot(2,4,3)
    plot(delay,TS_NMSE_Mean(:,1),'-k',delay,TS_NMSE_Mean(:,2),'-r',delay,TS_NMSE_Mean(:,3),'-m');
    xlabel('Data Points');
    ylabel('NMSE');
    subplot(2,4,4)
    plot(delay,TS_NMSE_Std(:,1),'-k',delay,TS_NMSE_Std(:,2),'-r',delay,TS_NMSE_Std(:,3),'-m');
    xlabel('Data Points');
    ylabel('NMSE Std');
    subplot(2,4,5)
    plot(delay,TS_NDEI_Mean(:,1),'-k',delay,TS_NDEI_Mean(:,2),'-r',delay,TS_NDEI_Mean(:,3),'-m');
    xlabel('Data Points');
    ylabel('NDEI');
    subplot(2,4,6)
    plot(delay,TS_NDEI_Std(:,1),'-k',delay,TS_NDEI_Std(:,2),'-r',delay,TS_NDEI_Std(:,3),'-m');
    xlabel('Data Points');
    ylabel('NDEI');
    subplot(2,4,7)
    plot(delay,TS_AE_Mean(:,1),'-k',delay,TS_AE_Mean(:,2),'-r',delay,TS_AE_Mean(:,3),'-m');
    xlabel('Data Points');
    ylabel('AE');
    subplot(2,4,8)
    plot(delay,TS_AE_Std(:,1),'-k',delay,TS_AE_Std(:,2),'-r',delay,TS_AE_Std(:,3),'-m');
    xlabel('Data Points');
    ylabel('AE Std');
    suptitle(strcat(dataName,'-Error Summary'));
    legend('Location','EastOutside','Neurons 10','Neurons 20','Neurons 30');
    saveas(gcf,strcat(path,dataName,'-TS-ErrorSummary','.png'));

    % Save variables from instance for later
    save(strcat(path,dataName,'-project3-TS-variables','.mat'));
end    