% Script LogRegGD.m
#{
Author:   Seth Thor
Date:     08/03/2020
Revised:  
Description:
  This script performs logistic regression gradient descent on a 
  given dataset. Sample data on gmat, gpa, work_experience, and admission
  provided for X, y. 
  
  Need to initialize theta, iter, alpha, p  
#}

clear;

function [J, h] = CostFunction(theta, X, y);
  m = size(X, 2);                               % # of samples/columns
  h = 1./( 1 .+ exp(-X*theta) );                % hypothesis function
  J = ( 1/m ) * -y'*log(h) - (1-y)'*log(1-h);   % Cost function
endfunction

function final_theta = GD(iter, alpha, theta, X, y)
  m = size(X, 2);
  j = zeros(iter,1);
  for i = 1:iter
    h = 1./( 1 .+ exp(-X*theta) );
    theta = theta - (alpha/m) * X'*(h - y); % Careful with dimensions here
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
 X*theta = h: 1 x m(samples)
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

% Use p% of dataset for training
p = 0.25;

% range of data to use for training
train_len = length(gmat)*p; 

%Design matrix - set to training data
X = [ones(train_len, 1), gmat_norm(1:train_len), gpa(1:train_len), work_experience(1:train_len)];
y = admitted(1:train_len);

%Initialize theta
theta = [0;0;0;0];

% Perform gradient descent
alpha = 0.01; % alpha, learning rate
iter = 1000;  % iterations
final_theta = GD(iter, alpha, theta, X, y)

% Now use model to predict test set. Reset variables to test data
s = length(gmat); % # of total samples
X = [ones(s - train_len + 1, 1), gmat_norm(train_len:s), gpa(train_len:s), work_experience(train_len:s)];
y = admitted(train_len:s);
y_predicted = 1./(1 .+ exp(-X*final_theta));

% Convert predictions into boolean (admit: 1, denied: 0)
% and calculate accuracy of predictions
y_pbool = zeros(length(y_predicted),1);
match = 0;
for i=1:length(y)
  
  if ( y_predicted(i) < 0.5 )
    y_pbool(i) = 0;
  elseif ( y_predicted(i) >= 0.5 )
    y_pbool(i) = 1;
  endif
  
  if ( y_pbool(i) == y(i) )
    match = match + 1;
  endif
  
end

accuracy = match/length(y)
figure
plot([1:length(y_predicted)], y_pbool, 'x')
hold on 
plot([1:length(y_predicted)], y, 'o')
hold off


