function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
% Hong San Wong (hswong1@uci.edu)
%
% LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
% regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% h_theta(X)
% X is a 12x2 vector
% Y is a 12x1 vector
h = X*theta; %Result in a 12x1 vector
diff = h-y;
theta1 = [0; theta(2:end, :)];
% theta1 = [0 ; 1]
reg = theta1'*theta1;
J = (1/(2*m))*(diff'*diff) + (lambda/(2*m))*(reg);


% Regularized linear regression gradient
% X = 12x2 and diff = 12x1
% therefore: X'*diff = > 2x12 match 12x1
grad = (1/m)*(X'*diff) + (lambda/m)*(theta1);


% =========================================================================

grad = grad(:);

end
