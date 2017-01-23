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
display("hooya...........")
thetax = X * theta;
HX = sigmoid(thetax);

first_part_of_cost =  -(y' * log(HX));
second_part_of_cost = (1-y)' * log(1-HX);
regularization_term = (lambda/(2*m)) * sum(theta(2:length(theta)).^2);

J = ((1/m) * (first_part_of_cost - second_part_of_cost)) + regularization_term;

grad(1) = (1/m) * ((X(:,1)' * (HX - y)));

grad(2:length(grad)) = (1/m) * ( X(:,2:columns(X))' * (HX - y)) + ( (lambda/m) * theta(2:length(theta)));

grad
% =============================================================

end
