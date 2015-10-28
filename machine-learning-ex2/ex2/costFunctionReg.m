function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypothesis = sigmoid(X * theta); %hypothesis is a 100x1 matrix

temp1 = (-y)' * (log(hypothesis)); %1x100 * 100x1 = 1x1
temp2 = (1 - y)' * (log(1 - hypothesis)); %1x100 * 100x1 = 1x1
temp3 = (1 / m) * sum(temp1 - temp2);
temp4 = (lambda / (2 * m)) * sumsqr(theta(2: (size(theta, 1)), 1));

J = temp3 + temp4;

grad(1) = (1 / m) * sum((hypothesis - y));

for index = 2:size(theta)
	grad(index) = (((1 / m) * sum((hypothesis - y) .* X(:, index))) + ((lambda / m) * theta(index)));
end



% =============================================================

end