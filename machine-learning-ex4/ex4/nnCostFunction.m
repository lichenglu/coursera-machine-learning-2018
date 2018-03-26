function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
a_1 = [ones(m, 1) X]; % 5000x401 

z_2 = Theta1 * a_1'; % 25x5000 
a_2 = sigmoid(z_2'); % 5000x25 
a_2 = [ones(m, 1) a_2]; % 5000x26

size(a_2)

z_3 = Theta2 * a_2'; % 10x5000
a_3 = sigmoid(z_3); % 10x5000

Y = dummyvar(y); % 5000x10

% The dot product here is very important
j = (-Y .* log(a_3)') - (1 - Y) .* log(1 - a_3)';
J = sum(j(:)) / m; % cost without regularization

% regularization
reg1_matrix = (Theta1(:, 2:end) .^ 2);
reg2_matrix = (Theta2(:, 2:end) .^ 2);
reg = (sum(reg1_matrix(:)) + sum(reg2_matrix(:))) * (lambda / (2 * m));

J = J + reg;

% BACK PROPOGATION
d3 = a_3 - Y';  % has same dimensions as a3

d2 = (Theta2' * d3) .* [ones(1, size(z_2,2)); sigmoidGradient(z_2)]; % has same dimensions as a2

D1 = d2 * a_1; 
D1 = D1(2:end, :); % has same dimensions as Theta1
D2 = d3 * a_2;    % has same dimensions as Theta2

Theta1_grad = Theta1_grad + (1/m) * D1;
Theta2_grad = Theta2_grad + (1/m) * D2;


% REGULARIZATION OF THE GRADIENT
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*(Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*(Theta2(:,2:end));

% backprop
% step 1
% for t = 1:m
%     xi = X(t, :)';
%     yi = Y(t, :)';
%         
%     zi2 = Theta1 * xi; % 25 * 5000
%     ai2 = sigmoid(zi2); % 25 * 5000
%     ai2 = [1; ai2]; % 26 * 5000
%     
%     zi3 = Theta2 * ai2; % 10 * 5000
%     ai3 = sigmoid(zi3); % 10 * 5000
%     
%     % step 2
%     d3 = zeros(num_labels, 1);
%     for k = 1:num_labels
%         d3(k) = ai3(k) - yi(k);
%     end
%     % step 3
%     d2 = Theta2 * d3' .* [1; sigmoidGradient(zi2)];
%     
%     D1 = d2(:,2:end)' * a_1;
%     D2 = d3' * a;
% end

% For loop version
% X = [ones(m, 1) X];
% Y = dummyvar(y);
% for i = 1:m
%     xi = X(i, :)';
%     yi = y(i, :)';
%         
%     z_2 = Theta1 * xi;
%     a_2 = sigmoid(z_2);
%     a_2 = [1; a_2];
%     
%     z_3 = Theta2 * a_2;
%     a_3 = sigmoid(z_3);
%     
%     for k = 1:num_labels
%         if (yi == k); y_k = 1; else; y_k = 0; end
%         if (y_k == 1); h = log(a_3); else; h = log(1 - a_3); end
%         
%         h_k = h(k);
%         
%         cost_k = -y_k * h_k - (1 - y_k) * h_k;
%         J = J + cost_k;
%     end
% end
% 
% J = J / m;
% 
% j1 = size(a_2, 1) - 1;
% k1 = size(X, 2);
% 
% j2 = size(Y, 2);
% k2 = size(a_2, 1);
% 
% reg1 = 0;
% for j = 1:j1
%     for k = 2:k1
%         reg1 = reg1 + Theta1(j, k)^2;
%     end
% end
% 
% reg2 = 0;
% for j = 1:j2
%     for k = 2:k2
%         reg2 = reg2 + Theta2(j, k)^2;
%     end
% end
% 
% reg = (reg1 + reg2) * (lambda / (2 * m));
% 
% J = J + reg;
 


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
