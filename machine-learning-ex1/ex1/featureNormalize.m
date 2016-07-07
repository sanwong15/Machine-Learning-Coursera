function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. 

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2)); %mean
sigma = zeros(1, size(X, 2)); %STD

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
     

for i=1:size(X, 2)
    mu(1,i) = mean(X(:,i));
    sigma(1,i) = std(X(:,i));
end


for j=1:size(X, 2)
    X_norm(:,j) = (X(:,j)-mu(j))/sigma(j);
end

% ============================================================

end