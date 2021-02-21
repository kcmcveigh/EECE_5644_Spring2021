% Kieran McVeigh 2/21/21
% Generate Data
mean_zero_vector = [-1; 1; -1; 1];
covariance_zero_matrix =[2 -.5 .3 0; -.5 1 -.5 0; 0.3 -0.5 1 0; 0 0 0 2];

mean_one_vector = [1; 1; 1; 1];
covariance_one_matrix =[1 .3 -.2 0; .3 2 .3 0; -0.2 0.3 1 0; 0 0 0 3];

rand_floats = sort(rand(10000,1));
class_zero = rand_floats(rand_floats < .7);
class_one = rand_floats(rand_floats >= .7);

class_zero_data = mvnrnd(mean_zero_vector',covariance_zero_matrix,length(class_zero));
class_one_data =mvnrnd(mean_one_vector',covariance_one_matrix,length(class_one));

class_zero_data = [class_zero_data zeros(length(class_zero),1)];
class_one_data = [class_one_data ones(length(class_one),1)];

all_data = [class_zero_data; class_one_data];

writematrix(all_data)



