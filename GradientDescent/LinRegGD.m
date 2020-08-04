% Script LinRegGD.m
#{
Author:   Seth Thor
Date:     08/01/2020
Revised:  08/03/2020
Description:
  This script performs linear regression gradient descent on a 
  given dataset. User needs to provide data file directories
  for X, y. Need to initialize theta, iter.
#}
clear;

function [train_set, test_set] = SplitData(p, data)
  
  m = length(data);
  nSample = floor(p*m); 
  rndIDX = randperm(m); % random permutation of indices
  
  train_set = data(rndIDX(1:nSample), :);
  test_set = data(rndIDX(nSample+1:m), :);  
  
endfunction

function [X_norm, y_norm] = FeatureNormalization(X, y)
  
  % Normalize using mean and stddev
  mu_X = mean(X);
  sigma_X = std(X);
  X_norm = (X - mu_X)./sigma_X;
  
  mu_y = mean(y);
  sigma_y = std(y);
  y_norm = (y - mu_y)./sigma_y;
  
endfunction

function y_predicted = h(theta, X)
  
  y_predicted = X*theta;
  
endfunction

function J = CostFunction(theta, X, y);
  m = length(y); % # of samples/columns
  h = X*theta;    % hypothesis function
  J = ( 1/(2*m) ) * sum( (h - y).^2 ); % Cost function
endfunction

function final_theta = GD(iter, alpha, theta, X, y)
  m = length(y);
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
Data file must have data in format [x0 x1 x2 ... | y]
After processing, the matrices will take the following dimensions:

 X: m(samples) x n(parameters)
 theta: n(parameters) x 1
 theta*X = h: m(samples) x 1
 y: m(samples) x 1
 
#}

#{
%Example:
data = [2 3 5 2 1;
        3 5 6 2 2;
        6 5 3 5 3];
After processing the data, 
X = [1 2 3 5 2;
     1 3 5 6 2;
     1 6 5 3 5];
y = [1; 2; 3];
theta = [0.1; 0.2; 0.1; 0.2; 0.1]; 
#}

%%%% DATA PROCESSING %%%%
% Load data
data = load('HousePrices.txt');

% Percent of data to use for training
p = 0.5; 

% Get dimensions of final design matrix, X
m = length(data); n = size(data, 2) - 1;

% Split the data
[train_set, test_set] = SplitData(p, data);

X_train = train_set(:, 1:n);
y_train = train_set(:, n+1);

[X_norm, y_norm] = FeatureNormalization(X_train, y_train);

% Add x0 column
X_norm = [ones(length(y_norm),1), X_norm];

%%%% TRAIN THE MODEL %%%%
a = 0.01; % alpha, learning rate
i = 500;  % iterations
theta = [1; 1; 1]; 
final_theta = GD(i, a, theta, X_norm, y_norm)

%%%% TEST THE MODEL %%%%
X_test = test_set(:, 1:n);
y_test = test_set(:, n+1);

[X_norm, y_norm] = FeatureNormalization(X_test, y_test);

% Add x0 column
X_norm = [ones(length(y_norm),1), X_norm];

% Predict y using test data, assess performance
y_predicted = h(final_theta, X_norm);
J = CostFunction(final_theta, X_norm, y_norm)

figure;
plot(y_predicted, y_norm, 'o')
hold on
plot([-1:0.01:1],[-1:0.01:1], '--')
hold off
xlabel('Predicted y'); ylabel('Test y');


