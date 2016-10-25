function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% Theta is given. a1 can be calculated by sigmoid(Theta1*X). Output(h) =
% a*Theta2

a1 = [ones(m,1) X]; % size of X: 5000*400 ; m = 5000 ; a1 becomes 5000*401
% With Theta1 = 25*401
z2 = a1*Theta1'; % 5000*25 (Theta1 is 25 by 401)
% With Theta2 = 10*26
a2 = [ones(size(z2, 1), 1) sigmoid(z2)]; % 5000*26
z3 = a2*Theta2';
a3 = sigmoid(z3);

[p_max, i_max]=max(a3, [], 2); % p_max is the max score while the i_max is the label 
p = i_max;
% =========================================================================


end
