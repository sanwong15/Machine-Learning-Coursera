function randomSelectedExamplePlot(X,y,Xval,yval,lambda,run)

m = size(X,1);

error_train = zeros(m,1);
error_val = zeros(m,1);


for l = 1:run
    % Randomly select data
    % Note
    % msize = numel(YourMatrix);
    % YourMatrix(randperm(msize, x))

    % OR
    
    % R = randperm(length(x));
    % indices = R(1:800);
    % y = x(indices);
    
    for i = 1:m
        % p = randperm(n) returns a random permutation of the integers 1:n
        sel = randperm(size(X,1));
        % In our case that m = 12.
        % sel = 6     3    11     7     8     5     1     2     4     9    10    12
        % sel is a 1 by 12 vector (aka 1 by m vector)
        sel = sel(1:i);
        
        %Create the matrix that contain the selected data
        X_sel = X(sel,:);
        y_sel = y(sel,:);
        
        % Learn the parameters using randomly selected Training data
        theta = trainLinearReg(X_sel,y_sel,lambda);
        % Evaluate with Theta got from Linear Reg (i.e. Calculate
        % error_train.)
        [J, grad] = linearRegCostFunction(X_sel, y_sel, theta, 0);
        
        % Update error_train
        error_train(i) = error_train(i) + J;
        
        % Evaulate with CV set (i.e: Calculate error_val)
        sel_val = randperm(size(Xval,1));
        sel_val = sel_val(1:i);
        X_sel = Xval(sel_val,:);
        y_sel = yval(sel_val,:);
        
        [J, grad_val] = linearRegCostFunction(X_sel,y_sel,theta,0);
        
        % Update error_val
        error_val(i) = error_val(i) + J;
    end
end

% Average
error_train = error_train./run;
error_val = error_val./run;

% Plot the result
plot(1:m, error_train, 1:m, error_val);
xlabel('Number of training examples');
ylabel('Error');
axis([0 13 0 100]);
legend('Train','Cross Validation');


end