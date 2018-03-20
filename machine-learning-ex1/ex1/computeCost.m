function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
halfM = (1 / (2 * m));
for iter = 1:m
    xi = X(iter, :);
    yi = y(iter, :);
    total = trace(theta * xi);
    J = J + ((total - yi) .^ 2);
end
J = halfM * J;



% =========================================================================

end
