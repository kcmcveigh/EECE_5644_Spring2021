clear;
%Kieran McVeigh
% ML Spring 2021
% Generate Data for question two
n_samples = [20, 200, 2000, 10000];

mean_01 = [3 0];
cov_01 = [2 0; 0 1];

mean_02 = [0 3];
cov_02 = [1 0; 0 2];

mean_1=[2 2];
cov_1=[1 0; 0 1];

prior_0 = .65;
prior_1 = .35;

for n_idx=1:length(n_samples)
    n_sample = n_samples(n_idx);
    rand_n = rand(n_sample,1);
    n_class_0 = (sum(rand_n<prior_0));
    n_class_1=n_sample-n_class_0;
    
    rand_n_01 = sum(rand(n_class_0,1)>.5);
    class_01_samples = mvnrnd(mean_01,cov_01,rand_n_01);
    class_01_samples(:,3)=0;
    
    class_02_samples = mvnrnd(mean_02,cov_02,(n_class_0-rand_n_01));
    class_02_samples(:,3)=0;
    
    class_0_samples = [class_01_samples; class_02_samples];
    
    class_1_samples = mvnrnd(mean_1,cov_1,n_class_1);
    class_1_samples(:,3)=1;
    
    all_samples = [class_0_samples; class_1_samples];
    writematrix(all_samples,strcat('question2_',string(n_sample))); 
end