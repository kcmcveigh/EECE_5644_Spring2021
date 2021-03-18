% Kieran McVeigh
% ML spring 2021
% inspiration taken from HW examples from Summer2_2020 take home examples
% from matlab
clear;
%% setup data
[xTrain,yTrain,xValidate,yValidate] = hw2q1;

xTrain = xTrain';
yTrain = yTrain';

xValidate = xValidate';
yValidate = yValidate';

x_validate_squared = xValidate .* xValidate;
x_validate_cubed = x_validate_squared .* xValidate;
x_validate_xsq_interaction = x_validate_squared.*flip(xValidate);
x_validate_x1x2_interaction = xValidate(:,1).*xValidate(:,2);
x_validate_cubic = [ones(1000,1) xValidate...
    x_validate_x1x2_interaction x_validate_squared...
    x_validate_xsq_interaction x_validate_cubed];

x_train_squared = xTrain .* xTrain;
x_train_cubed = x_train_squared .* xTrain;
x_train_xsq_interaction = x_train_squared.*flip(xTrain);
x_train_x1x2_interaction = xTrain(:,1).*xTrain(:,2);
x_train_cubic = [ones(100,1) xTrain...
    x_train_x1x2_interaction x_train_squared...
    x_train_xsq_interaction x_train_cubed];

q1_x_train_cubic = x_train_cubic; q1_x_validate_cubic = x_validate_cubic;
q1_y_train = yTrain;q1_y_validate=yValidate;

writematrix(q1_x_train_cubic);writematrix(q1_x_validate_cubic);
writematrix(q1_y_train);writematrix(q1_y_validate);
%% saved data from previous commented out step so wouldn't have to keep regnerating
load('q1_x_train_cubic.txt');load('q1_x_validate_cubic.txt');
load('q1_y_train.txt');load('q1_y_validate.txt');
%% fit weights ML
psuedo_inv = inv(q1_x_train_cubic'*q1_x_train_cubic)*q1_x_train_cubic';
weights_ml =psuedo_inv*q1_y_train;
%% predict
y_hat_ml = weights_ml'*q1_x_validate_cubic';
squared_error_ml = CalcSquaredError(y_hat_ml,q1_y_validate);

%% MAP
lambda_values = logspace(-7,4,1000);
for lambda_idx=1:length(lambda_values)
    prior_cov = lambda_values(lambda_idx)*eye(10);
    map_inv = inv(q1_x_train_cubic'*q1_x_train_cubic +inv(prior_cov))*q1_x_train_cubic';
    weights_map =map_inv*q1_y_train;
    y_hat_map = weights_map'*q1_x_validate_cubic';
    squared_error_map(lambda_idx) = CalcSquaredError(y_hat_map,q1_y_validate);
end

%% get min map estimates;
[min_map_sq_err min_lambda_idx]=min(squared_error_map);
prior_cov = lambda_values(min_lambda_idx)*eye(10);
map_inv = inv(q1_x_train_cubic'*q1_x_train_cubic +inv(prior_cov))*q1_x_train_cubic';
weights_map =map_inv*q1_y_train;




x1 = [1 2 4 6];
x2 =[1 3 5 7];
y_hat_map_x1 = weights_map'*q1_x_validate_cubic';
y_hat_ml_x1 = weights_ml'*q1_x_validate_cubic';

plot(lambda_values,squared_error_map)
hold on
yline(squared_error_ml)
hold off


weights_ml
% hold on
% 
% scatter(q1_x_validate_cubic(:,2),y_hat_ml_x1,'+')
% scatter(q1_x_validate_cubic(:,2),y_hat_map_x1)
% scatter(q1_x_validate_cubic(:,2),q1_y_validate)
% hold off





function squared_error = CalcSquaredError(y_hat,y_obs)
    error = y_obs - y_hat';
    squared_error = mean(error.*error);
end



