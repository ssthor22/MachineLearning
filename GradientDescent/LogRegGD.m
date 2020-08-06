% Script LogRegGD.m
#{
Author:   Seth Thor
Date:     08/03/2020
Revised:  08/05/2020
Description:
  This script performs logistic regression gradient descent on a 
  given dataset. Sample data on gmat, gpa, work_experience, and admission
  provided for X, y. 
  
  Need to initialize theta, iter, alpha, p  
#}

clear;

function data_norm = FeatureNormalization(data)
  
  % Normalize using mean and stddev
  range_data = max(data) - min(data);
  data_norm = data./range_data;
  
endfunction

function [train_set, test_set] = SplitData(p, data)
  
  m = length(data);
  nSample = floor(p*m); 
  rndIDX = randperm(m); % random permutation of indices
  
  train_set = data(rndIDX(1:nSample), :);
  test_set = data(rndIDX(nSample+1:m), :);  
  
endfunction

function [J, h] = CostFunction(theta, X, y);
  
  m = length(y);  % # of samples/columns
  h = sigmoid(X*theta); % hypothesis function
  J = ( 1/m ) * -y'*log(h) - (1-y)'*log(1-h); % Cost function

endfunction

function final_theta = GD(iter, alpha, theta, X, y)
  
  m = length(y);
  j = zeros(iter,1);
  for i = 1:iter
    h = sigmoid(X*theta);
    theta = theta - (alpha/m) * X'*(h - y); % Careful with dimensions here
    j(i) = CostFunction(theta, X, y);
  end
  final_theta = theta;
  plot([1:iter], j)
  xlabel('Iterations'); ylabel('Cost Function, J');
  
endfunction

function g = sigmoid(z)

% Computes the sigmoid of z, where z = X*theta
  g = 1./( 1 .+ exp(-z) );
  
endfunction

function accuracy = ModelPerformance(theta, X, y)
  
  z = X*theta;
  y_predicted = sigmoid(z);

  % Convert predictions into boolean (admit: 1, denied: 0)
  % and calculate accuracy of prediction
  y_pbool = zeros(length(y_predicted),1);
  match = 0;
  for i=1:length(y)
    
    if ( y_predicted(i) < 0.5 )
      y_pbool(i) = 0;
    elseif ( y_predicted(i) >= 0.5 )
      y_pbool(i) = 1;
    endif
    
  endfor
    
  accuracy = mean(double(y_pbool == y));

endfunction

#{
Data file must have data in format [x0 x1 x2 ... | y]
After processing, the matrices will take the following dimensions:

 X: m(samples) x n(parameters)
 theta: n(parameters) x 1
 theta*X = h: m(samples) x 1
 y: m(samples) x 1
 
#}


% Test dataset of admittance based on gmat, gpa, work experience
% Taken from: https://datatofish.com/logistic-regression-python/
gmat = [780; 750; 690; 710; 680; 730; 690; 720; 740; 690; 610; 690; 710; 680; 
        770; 610; 580; 650; 540; 590; 620; 600; 550; 550; 570; 670; 660; 580; 
        650; 660; 640; 620; 660; 660; 680; 650; 670; 580; 590; 690];
        
% Normalize the gmat values to avoid distortion        
gmat_norm = gmat/( max(gmat) - min(gmat) );
gpa = [4;3.9;3.3;3.7;3.9;3.7;2.3;3.3;3.3;1.7;2.7;3.7;3.7;3.3;3.3;3;2.7;3.7;
       2.7;2.3;3.3;2;2.3;2.7;3;3.3;3.7;2.3;3.7;3.3;3;2.7;4;3.3;3.3;2.3;2.7;
       3.3;1.7;3.7];
work_experience = [3;4;3;5;4;6;1;4;5;1;3;5;6;4;3;1;4;6;2;3;2;1;4;1;2;6;4;2;6;5;
                   1;2;4;6;5;1;2;1;4;5];
admitted = [1;1;0;1;0;1;0;1;1;0;0;1;1;0;1;0;0;1;0;0;1;0;0;0;0;1;1;0;1;1;0;0;1;
            1;1;0;0;0;0;1];
            
data = [gmat, gpa, work_experience, admitted];

%data = load('filename.txt');
data_norm = FeatureNormalization(data);
data_norm = [ones(length(data_norm),1), data_norm];

m = length(data_norm); 
n = size(data_norm, 2) - 1;

% Use p% of dataset for training
p = 0.25;
[train_set, test_set] = SplitData(p, data_norm);

%Design matrix - set to training data
X = train_set(:,1:n);
y = train_set(:,n+1);

%Initialize theta
theta = [0;0;0;0];

% Perform gradient descent
alpha = 0.01; % alpha, learning rate
iter = 10000;  % iterations
final_theta = GD(iter, alpha, theta, X, y)

% Now use model to predict test set. Reset variables to test data
X = test_set(:,1:n);
y = test_set(:,n+1);

accuracy = ModelPerformance(final_theta, X, y)


