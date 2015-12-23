function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); %number of observations
num_labels = size(Theta2, 1); % number of possible outcomes

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

%add a column of 1's to the X matrix
X = [ones(m, 1) X];
%compute hidden layer input values, aka a(2)
ztwo=Theta1*transpose(X)
atwo= 1.0 ./ (1.0 + exp(-ztwo))
%add the bias term to a(2)
atwo=[ones(1,m);atwo]

%compute output layer values, aka a(3)
zthree=Theta2*atwo
athree=1.0 ./ (1.0 + exp(-zthree))
% each column in athree represents outcome vector of probabilities for 1
% obs.
%use max function to find the label of each obs.
[Y,I]=max(athree)
p=transpose(I)


% =========================================================================


end
