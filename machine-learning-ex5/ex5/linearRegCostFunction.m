function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
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

t4=(X*theta)-y;
t=t4.^2;
t1=theta;
t1(1)=0;
t1=t1.^2;
J=(sum(t)+(lambda*sum(t1)))/(2*m);

grad=(X'*t4)/m;
t2=theta;
t2(1)=0;
grad=grad+(lambda*t2)/m;















% =========================================================================

grad = grad(:);

end
