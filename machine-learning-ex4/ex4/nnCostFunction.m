function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% Hong San Wong (hswong1@uci.edu)
% NNCOSTFUNCTION Implements the neural network cost function for a two layer
% neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

% nn_params = [Theta1(:) ; Theta2(:)];
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1 (Feedforward)
% y is a 5000x1 label vector (class: 1 - 10)
% First, unroll it so that each label is a 10x1 vector
I = eye(num_labels);
Y = zeros(m, num_labels); % m by num_labels => 5000x10
for i=1:m
    Y(i,:) = I(y(i),:);
end

% Explain:
% eye gives the following: let's say it's a 3x3
% I = 1 0 0
%     0 1 0
%     0 0 1
%    
% y(i) = label 2 (which means y(i) = 2)
% Y(i,:) = I(2,:) => 0 1 0

% Feedforward
% X matrix: m by num_class
A1 = [ones(m,1) X]; % adding one (bias)
Z2 = A1*Theta1';
A2 = [ones(size(Z2,1), 1) sigmoid(Z2)]; % adding one (bias)
Z3 = A2*Theta2';
A3 = sigmoid(Z3);

% Calculate the cost function
reg = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
J = (1/m)*sum(sum(-Y.*log(A3)-(1-Y).*log(1-A3)))+reg;

% Backpropagation
% Calculate delta
delta3 = A3-Y;
delta2 = (delta3*Theta2).*sigmoidGradient([ones(size(Z2,1),1) Z2]);
delta2 = delta2(:,2:end);

% Calculate partial derivative (i.e: Cap delta Vector) by delta
% accumulated within the same layer
Delta2 = delta3'*A2;
Delta1 = delta2'*A1;

% Calculate Theta gradient: i.e D vector = dJ/dTheta
% Add zero vector: Theta_size by 1 because there is no regularization is
% zero for j=0 terms
% Divide the accumulated gradient by m
Theta1_grad = (1/m)*Delta1+(lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad = (1/m)*Delta2+(lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
