% Kieran McVeigh 2/21/21
% Generate Data
clear;
mean_one_vector = [-1; 1; -1];
covariance_one_matrix =[1 0 0; 0 1.5 0.4; 0 0.4 .6];
eig(covariance_one_matrix)

mean_two_vector = [1; 1; 1];
covariance_two_matrix =[1.5 0.4 0; 0.4 1 0; 0 0 .7];
eig(covariance_two_matrix)

mean_three_vector = [1; 1; -1];
covariance_three_matrix =[.6 0 0.25; 0 1 0; 0.25 0 1];
eig(covariance_three_matrix)

mean_four_vector = [-1; -1; -1];
covariance_four_matrix =[.5 0 0.25; 0 1 0; 0.25 0 1];
eig(covariance_four_matrix)

n_samples = [100,200,500,1000,2000,5000,100000];

for n_sample_idx=1:length(n_samples)
    n_sample = n_samples(n_sample_idx);
    rand_floats = sort(rand(n_sample,1));
    
    
    class_one = rand_floats(rand_floats < .25);
    class_two = rand_floats(rand_floats < .5 & rand_floats >=.25);
    class_three = rand_floats(rand_floats < .75 & rand_floats >=.5);
    class_four = rand_floats(rand_floats >=.75);

    class_one_data = mvnrnd(mean_one_vector',covariance_one_matrix,length(class_one));
    class_two_data =mvnrnd(mean_two_vector',covariance_two_matrix,length(class_two));
    class_three_data = mvnrnd(mean_three_vector',covariance_three_matrix,length(class_three));
    class_four_data =mvnrnd(mean_four_vector',covariance_four_matrix,length(class_four));

    class_one_data = [class_one_data ones(length(class_one),1)];
    class_two_data = [class_two_data 2*ones(length(class_two),1)];
    class_three_data = [class_three_data 3*ones(length(class_three),1)];
    class_four_data = [class_four_data 4*ones(length(class_four),1)];

    all_data = [class_one_data; class_two_data; class_three_data; class_four_data];
    
    %writematrix(all_data,strcat(string(n_sample),'_question_data'));
end


scatter3(class_one_data(:,1),class_one_data(:,2),class_one_data(:,3));
hold on
scatter3(class_two_data(:,1),class_two_data(:,2),class_two_data(:,3),5,'r');
scatter3(class_three_data(:,1),class_three_data(:,2),class_three_data(:,3),5,'c');
scatter3(class_four_data(:,1),class_four_data(:,2),class_four_data(:,3),'o');
hold off 

%writematrix(all_data,strcat(string(n_sample),'_question_data'));



