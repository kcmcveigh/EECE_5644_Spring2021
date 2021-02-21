%% Question2 GenerateData 
% Kieran McVeigh 2/21/21
clear;

mean_one_vector = [1 0 0];
mean_two_vector =[1 1 1];
mean_threeA_vector = [0 1 0];%3a
mean_threeB_vector = [0 0 1];%3b

covariance = [.75 0 0; 0 .75 0; 0 0 .75];

%Set class priors to 0.3,0.3,0.4.
rand_nums = rand(10000,1);
class_1 = rand_nums < .3; n_class_1 = sum(class_1);
class_2 = (.3 <= rand_nums)& (rand_nums < .6); n_class_2 = sum(class_2);
class_3a = (.6 <= rand_nums)& (rand_nums < .8); n_class_3a = sum(class_3a);
class_3b =  .8 <= rand_nums; n_class_3b = sum(class_3b);

class_1_data = mvnrnd(mean_one_vector',covariance,n_class_1);
class_2_data =mvnrnd(mean_two_vector',covariance,n_class_2);
class_3a_data =mvnrnd(mean_threeA_vector',covariance,n_class_3a);
class_3b_data =mvnrnd(mean_threeB_vector',covariance,n_class_3b);

% scatter3(class_1_data(:,1),class_1_data(:,2),class_1_data(:,3))
% hold on
% scatter3(class_2_data(:,1),class_2_data(:,2),class_2_data(:,3))
% scatter3(class_3a_data(:,1),class_3a_data(:,2),class_3a_data(:,3))
% scatter3(class_3b_data(:,1),class_3b_data(:,2),class_3b_data(:,3))

class_1_data = [class_1_data, ones(n_class_1,1)];
class_2_data = [class_2_data, ones(n_class_2,1)*2];
class_3a_data = [class_3a_data, ones(n_class_3a,1)*3];
class_3b_data = [class_3b_data, ones(n_class_3b,1)*3];

all_data_question_two = [class_1_data; class_2_data; class_3a_data; class_3b_data];
label_marker = ['o','x','^'];
for label=1:3
    
    class_data = all_data_question_two(all_data_question_two(:,4)==label,:);
    scatter3(class_data(:,1),...
        class_data(:,2),...
        class_data(:,3),...
        ones(length(class_data),1)*5,...
        label_marker(label));
    hold on
end
writematrix(all_data_question_two)

