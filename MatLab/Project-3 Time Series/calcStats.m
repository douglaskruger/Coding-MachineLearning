function [stats] = calcStats(targets, output)
% ***********************************************************************
% (c) 2014 Douglas Kruger
% ECE 626 - Advanced Neural Networks
% ***********************************************************************
    errors = gsubtract(targets,output);
    n=size(targets,2);
    sd=std(targets);
    
    mse=0;
    nmse=0;
    ae=0;
    for i = 1:n
        mse  = mse + errors(i)^2/n;
        nmse = nmse + errors(i)^2;
        ae = ae + abs(errors(i))/n;
    end
    nmse = nmse / (sd^2 * n);
    ndei = sqrt(mse)/sd;
    
    stats(1)=mse;
    stats(2)=nmse;
    stats(3)=ndei;
    stats(4)=ae;
end