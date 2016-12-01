function Z = projectData(X, U, K)
%Hong San Wong (hswongg1@uci.edu)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Reduce from N to K
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.

%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K); %m*K

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

 U_reduce = U(:,1:K); %m*K
 for i=1:size(X,1) % m loop
    x = X(i, :)'; % x = (1*N)' = (N*1)
    
    for k=1:K
%        projection_k = x(i,k) * U_reduce(i,k);
         projection_k = x' * U_reduce(:, k);
        Z(i,k) = projection_k;
    end
    
 end


% =============================================================

end
