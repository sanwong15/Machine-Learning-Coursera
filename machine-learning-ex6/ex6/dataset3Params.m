function [C, sigma] = dataset3Params(X, y, Xval, yval)
% Hong San Wong (hswong1@uci.edu)
%
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%Input: X y Xval yval Output: C sigma

testValues = [0.01 0.03 0.1 0.3 1 3 10 30];
minError = Inf;

for i=1:length(testValues)
    for j=1:length(testValues)
        
        C_val = testValues(i);
        sigma_val = testValues(j);
        
        model= svmTrain(X, y, C_val, @(x1, x2) gaussianKernel(x1, x2, sigma_val));
        predictions = svmPredict(model, Xval); %returns a vector of predictions using a trained SVM model
        error_cv = mean(double(predictions ~= yval));
        
        % Amoung all the error_cv, return the (C, sigma) pair that has the smallest
        % error_cv
        if error_cv < minError
            minError = error_cv;
            C = C_val;
            sigma = sigma_val;
        end
        
    end
end



% =========================================================================

end
