function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
% X is the training data vector
% y is the correct answer for the training data vector
 
J = 0;
%J(theta) = (1/2m)SumOverM(h(x)-y)^2
for i=1:m
    hX = theta(1)*X(i,1)+theta(2)*X(i,2);
    diff = hX - y(i);
    diffSquare = diff^2;
    J = J+diffSquare;
end

J = J/(2*m);

% =========================================================================

end
