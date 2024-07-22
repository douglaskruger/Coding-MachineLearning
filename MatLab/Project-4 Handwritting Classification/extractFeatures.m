%% ***************************************************************************************
% ECE 623 - Data Exploration and Evolutionary Computing
% University of Alberta
% (c) 2014 Douglas Kruger
% ***************************************************************************************
function [classBinary, features]=extractFeatures(path, iFile, oFile)
%clear all; close all;
%path='C:\ECE623\project\';
%iFile='train';
%oFile='features';

%clear all; close all;
inputFile=strcat(path,iFile,'.csv');
outputFile=strcat(path,oFile,'.csv');
averageFile=strcat(path,oFile,'-average.csv');

% Define Constants
charData=csvread(inputFile,1); % Read the original char data
charWidth=28;  % Width in bits
charHeight=28; % Height in bits
threshold=50; % Bit threshold for being on/off
charElement=size(charData,1);
% The class of the data is in column 1
charClass=charData(:,1);
% Remove the class from the dataset
charData(:,1)=[];
features(charElement,16)=zeros;

% Create array - (imageRow,characterType,hSize,vSize)
pixelXY=zeros(charElement,charWidth, charHeight);

% Process all the data set rows
%for dataRow = 1:10
featuresAvg(10,16)=zeros;
classBinary(charElement,10)=zeros;
for dataRow = 1:charElement
    % Set the binary class element
    classBinary(dataRow,charClass(dataRow,1)+1)=1;
    
    % Convert the initial single data row to a 28x28 matrix
    % Then invert it and apply threshold and finally flip upside down (y
    % co-ordinates
    pixelXY(dataRow,:,:)=flipud(reshape(charData(dataRow,:,:),charWidth, charHeight)'>threshold);
    clear x y;
    [y,x]=find(reshape(pixelXY(dataRow,:,:),charWidth, charHeight));
    
    % Reset the counters
    xEdge=0;
    yEdge=0;
    xEdgeYSum=0;
    yEdgeXSum=0;
   
    % Walk through the image area
    for xIndex=min(x):max(x)
        for yIndex=min(y):max(y)
            % Check for the pixel being on and comparing on the edget of
            % the image
            if (pixelXY(dataRow,yIndex,xIndex)==1)
                if (xIndex==min(x))
                    xEdge=xEdge+1;
                    xEdgeYSum=xEdgeYSum+yIndex;
                end
                if (yIndex==min(y))
                    yEdge=yEdge+1;
                    yEdgeXSum=yEdgeXSum+xIndex;
                end
            % The pixel is off
            % Check if the next one (right/up) is on - then edge
            else
                if (yIndex<max(y)&&(pixelXY(dataRow,yIndex+1,xIndex)==1))
                    xEdge=xEdge+1;
                    xEdgeYSum=xEdgeYSum+yIndex;
                end
                if (xIndex<max(x)&&(pixelXY(dataRow,yIndex,xIndex+1)==1))
                    yEdge=yEdge+1;
                    yEdgeXSum=yEdgeXSum+xIndex;
                end
            end
        end
    end
    % Define the features
    % A1: Horizontal position - left edge of image to center of box
    % A2: Vertical position - bottom edge of image to center of box
    % A3: Width of box
    % A4: Height of box
    % A5: Total number of pixels on for character
    % A6: Mean horizontal position relative to center of box / width of box
    % A7: Mean vertical position relative to center of box / width of box
    % A8: Mean Squared horizontal position relative to center of box / width of box
    % A9: Mean squared vertical position relative to center of box / width of box
    % A10: Mean product of horizontal and vertical of each on pixel
    % A11: Mean squared horizontal times vertical of each on pixel
    % A12: Mean squared vertical times horizontal of each on pixel
    % A13: Mean number of vertical edges -- left to right
    % A14: Sum of vertical edges times position
    % A15: Mean number of horizontal edges -- bottom to top
    % A16: Sum of horizontal edges times position
    features(dataRow,1)=round((max(x)+min(x))/2);
    features(dataRow,2)=round((max(y)+min(y))/2);
    features(dataRow,3)=max(x)-min(x)+1;
    features(dataRow,4)=max(y)-min(y)+1;
    features(dataRow,5)=size(x,1);
    features(dataRow,6)=mean(x-features(dataRow,1))/features(dataRow,3);
    features(dataRow,7)=mean(y-features(dataRow,2))/features(dataRow,4);
    features(dataRow,8)=mean((x(:,1)-features(dataRow,1)).^2);
    features(dataRow,9)=mean((y(:,1)-features(dataRow,2)).^2);
    features(dataRow,10)=mean((x(:,1)-features(dataRow,1)).*(y(:,1)-features(dataRow,2)));
    features(dataRow,11)=mean((x(:,1)-features(dataRow,1)).^2.*(y(:,1)-features(dataRow,2)));
    features(dataRow,12)=mean((y(:,1)-features(dataRow,2)).^2.*(x(:,1)-features(dataRow,1)));
    features(dataRow,13)=xEdge/size(x,1);
    features(dataRow,14)=xEdgeYSum;
    features(dataRow,15)=yEdge/size(y,1);
    features(dataRow,16)=yEdgeXSum;
    featuresAvg(charClass(dataRow)+1,:)=featuresAvg(charClass(dataRow)+1,:)+features(dataRow,:);
    
end;

%csvwrite('C:\ECE623\kaggle\pixelsXY.csv',reshape(pixelXY(dataRow,:,:),charWidth, charHeight)); % Write the data
outputFile=strcat(path,oFile,'-hc.csv');
csvwrite(outputFile,[charClass features]); % Write the data - vertical
outputFile=strcat(path,oFile,'-vc.csv');
csvwrite(outputFile,[features]'); % Write the data - vertical
outputFile=strcat(path,oFile,'-h.csv');
csvwrite(outputFile,[features]); % Write the data - vertical
outputFile=strcat(path,oFile,'-v.csv');
csvwrite(outputFile,[features]'); % Write the data - vertical


outputFile=strcat(path,'output-v.csv');
csvwrite(outputFile,[classBinary]'); % Write the data - vertical    
    
% Determine the average feature values for the class
for class = 1:10
    featuresAvg(class,:)=featuresAvg(class,:)/size(find(charClass(:)==class-1),1);
end
outputFile=strcat(path,'features-average.csv');
class=[0 1 2 3 4 5 6 7 8 9];
csvwrite(averageFile,[class' featuresAvg]); % Write the data
end
