function idx = findClosestCentroids(X, centroids)
% Hong San Wong (hswong1@uci.edu)
% FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.

 for xi = 1:size(X,1) % For each data point
    min_d = Inf; % Init
    for k=1:K %Compare with each centroids
        x = X(xi,:);
        mu = centroids(k, :);
        diff = x'-mu'; %Square distance between the data point and centroids
        sqt_diff = diff'*diff;
        
        %If we find a closer centroids, we update the values of its class
        if(sqt_diff<min_d)
            min_d = sqt_diff;
            idx(xi) = k;
        end
    end
end



% =============================================================

end

