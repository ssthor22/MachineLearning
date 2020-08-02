% Script GradientDescent.m
#{
Author:   Seth Thor
Date:     08/01/2020
Revised:  
Description:
  This script performs linear regression gradient descent on a 
  given dataset. User needs to provide data file directories
  for X, y. Need to initialize theta, iter, 
#}
1;

function J = CostFunction(theta, X, y);
  m = size(X, 2); % # of samples/columns
  h = X*theta;    % hypothesis function
  J = ( 1/(2*m) ) * sum( (h - y).^2 ); % Cost function
endfunction

function final_theta = GD(iter, alpha, theta, X, y)
  m = size(X, 2);
  j = zeros(iter,1);
  for i = 1:iter
    h = X*theta;
    theta = theta - (alpha/m) * ((h - y)'*X)'; % Careful with transposes here
    j(i) = CostFunction(theta, X, y);
  end
  final_theta = theta;
  plot([1:iter], j)
  xlabel('Iterations'); ylabel('Cost Function, J');
endfunction

#{
Data must be in format:
 X: m(samples) x n(parameters)
 theta: n(parameters) x 1
 theta*X = h: m(samples) x 1
 y: m(samples) x 1
#}

#{
%Example:
X = [1 2 3 5 2;
     1 3 5 6 2;
     1 6 5 3 5];
theta = [0.1; 0.2; 0.1; 0.2; 0.1]; 
y = [1; 2; 3];
#}

a = 0.01; % alpha, learning rate
i = 500;  % iterations
final_theta = GD(i, a, theta, X, y)

