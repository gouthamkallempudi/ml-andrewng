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

k=sigmoid(X*theta);
l1=log(k);
l2=log(1-k);

sum1=(-y).*l1;
sum2=(1-y).*l2;
f=sum1-sum2;
w1=theta;
w1(1)=0;
w=w1.^2;
J=sum(f)/m  +   (lambda*sum(w))/(2*m);

a=sigmoid(X*theta)-y;
i=theta;
i(1)=0
grad=(X'*a) + lambda*i;

grad=grad/m;









% =============================================================

end
