function [TPR,FPR,accuracy] = calcPositiveStats(oValidation,tValidation,loop)
    % determine the class (use the index level) for the output and target
    oValidationClass = vec2ind(oValidation);
    tValidationClass = vec2ind(tValidation);
    [classes, instances] = size(oValidation);
    
    % Initialize variables
    TP(classes) = 0;
    TN(classes) = 0;
    FP(classes) = 0;
    FN(classes) = 0;
    TC=0;

    % Determine the True Positive, True Negative, False Positive, False
    % Negative
    for innerLoop = 1:instances
        for class = 1:classes
            if (oValidationClass(innerLoop) == class) 
                if (tValidationClass(innerLoop) == class)
                    TP(class) = TP(class) + 1;
                elseif (tValidationClass(innerLoop) ~= class)
                    FN(class) = FN(class) + 1;
                end
            elseif (oValidationClass(innerLoop) ~= class)
                if (tValidationClass(innerLoop) ~= class)
                    TN(class) = TN(class) + 1;
                elseif (tValidationClass(innerLoop) == class)
                    FP(class) = FP(class) + 1;
                end
            end
        end
        % Keep track of correct predictions
        if (oValidationClass(innerLoop) == tValidationClass(innerLoop))
            TC = TC + 1;
        end
    end
    % Accuracy = Total correct / Total instances
    accuracy = TC / instances;    
    
    % Evaluate all classes for each loop
    % need to do this outside of the previous loop
    for class = 1:classes
        TPR(class) = TP(class) / (TP(class) + FN(class));
        FPR(class) = FP(class) / (FP(class) + TN(class));
    end