% ***********************************************************************
% (c) 2014 Douglas Kruger
% ECE 626 - Advanced Neural Networks
% ***********************************************************************

% Loop through all the crossvalidation folds
close all; clear all; clc; format compact;

% Configuration
path='C:\ECE626\';
%path='C:\Users\kruger.WILLOWGLEN\Google Drive\MBA\ECE 626 - Advanced Neural\Project3aData\';
spread = 1;
neuron = [10 20 30];

for datasetLoop=1:2
    if datasetLoop==1
        % Mackey-Glass (m=14,d=10)
        delay=[6 10 12 14];
        m=[9 12 15];
        dataName='Mackey-Glass'
    else
        % SantaFe LaserA (m=8,d=3)
        delay=[2 3 5 10];
        m=[6 8 10];
        dataName='SantaFeLaserA';
    end
    for neuronLoop = 1:size(neuron,2)
        for delayLoop = 1:size(delay,2)
            for mLoop = 1:size(m,2)
                Errors=runRBFN(path, dataName, delay(delayLoop), m(mLoop), 1, neuron(neuronLoop));
                TS_MSE_Mean(delayLoop,neuronLoop,mLoop)=mean(Errors(1,:));
                TS_MSE_Std(delayLoop,neuronLoop,mLoop)=std(Errors(1,:));
                TS_NMSE_Mean(delayLoop,neuronLoop,mLoop)=mean(Errors(2,:));
                TS_NMSE_Std(delayLoop,neuronLoop,mLoop)=std(Errors(2,:));
                TS_NDEI_Mean(delayLoop,neuronLoop,mLoop)=mean(Errors(3,:));
                TS_NDEI_Std(delayLoop,neuronLoop,mLoop)=std(Errors(3,:));
                TS_AE_Mean(delayLoop,neuronLoop,mLoop)=mean(Errors(4,:));
                TS_AE_Std(delayLoop,neuronLoop,mLoop)=std(Errors(4,:));
            end
        end
    end

    % Plots - Actual, Predictions, Errors
    for mLoop = 1:size(m,2)
        x=linspace(1,delay(delayLoop),delay(delayLoop))
        figure
        subplot(4,2,1)
        plot(delay,TS_MSE_Mean(:,1,mLoop),'-k',delay,TS_MSE_Mean(:,2,mLoop),'-r',delay,TS_MSE_Mean(:,3,mLoop),'-m');
        xlabel('Delay');
        ylabel('MSE');
        subplot(4,2,2)
        plot(delay,TS_MSE_Std(:,1,mLoop),'-k',delay,TS_MSE_Std(:,2,mLoop),'-r',delay,TS_MSE_Std(:,3,mLoop),'-m');
        xlabel('Delay');
        ylabel('MSE Std');
        subplot(4,2,3)
        plot(delay,TS_NMSE_Mean(:,1,mLoop),'-k',delay,TS_NMSE_Mean(:,2,mLoop),'-r',delay,TS_NMSE_Mean(:,3,mLoop),'-m');
        xlabel('Delay');
        ylabel('NMSE');
        subplot(4,2,4)
        plot(delay,TS_NMSE_Std(:,1,mLoop),'-k',delay,TS_NMSE_Std(:,2,mLoop),'-r',delay,TS_NMSE_Std(:,3,mLoop),'-m');
        xlabel('Delay');
        ylabel('NMSE Std');
        subplot(4,2,5)
        plot(delay,TS_NDEI_Mean(:,1,mLoop),'-k',delay,TS_NDEI_Mean(:,2,mLoop),'-r',delay,TS_NDEI_Mean(:,3,mLoop),'-m');
        xlabel('Delay');
        ylabel('NDEI');
        subplot(4,2,6)
        plot(delay,TS_NDEI_Std(:,1,mLoop),'-k',delay,TS_NDEI_Std(:,2,mLoop),'-r',delay,TS_NDEI_Std(:,3),'-m');
        xlabel('Delay');
        ylabel('NDEI');
        subplot(4,2,7)
        plot(delay,TS_AE_Mean(:,1,mLoop),'-k',delay,TS_AE_Mean(:,2,mLoop),'-r',delay,TS_AE_Mean(:,3,mLoop),'-m');
        xlabel('Delay');
        ylabel('AE');
        subplot(4,2,8)
        plot(delay,TS_AE_Std(:,1,mLoop),'-k',delay,TS_AE_Std(:,2,mLoop),'-r',delay,TS_AE_Std(:,3,mLoop),'-m');
        xlabel('Delay');
        ylabel('AE Std');
        suptitle(strcat(dataName,'-Error Summary'));
        legend('Location','EastOutside','Neurons 10','Neurons 20','Neurons 30');
        saveas(gcf,strcat(path,dataName,'-RBF-ErrorSummary-m',num2str(m(mLoop)),'.png'));
    end
    
    % Save variables from instance for later
    save(strcat(path,dataName,'-project3-RBF-variables','.mat'));
end    