%Kieran McVeigh
% ML Spring 2021
% Question 2 part 1
clear;
mean_01 = [3 0];
cov_01 = [2 0; 0 1];
mean_02 = [0 3];
cov_02 = [1 0; 0 2];

% mean_0 = .5*mean_01 + .5*mean_02;
% cov_0 = .5*cov_01 + .5*cov_02;

mean_1=[2 2];
cov_1=[1 0; 0 1];

prior_0 = .65;
prior_1 = .35;

load('Question2Data/question2_10000.txt');
data = question2_10000(:,1:2);
labels =  question2_10000(:,3);

cov_0 =cov(data(labels==0,:));
like_0 = .5*mvnpdf(data,mean_01,cov_01)+.5*mvnpdf(data,mean_02,cov_02);
like_1 =mvnpdf(data,mean_1,cov_1);

like_ratio = like_1./like_0;

%% empirical gamma
gamma_values = linspace(0,100,10000);
optimal_error_rate = 1;
optimal_gamma = 0;
for gamma_idx=1:length(gamma_values)
    gamma = gamma_values(gamma_idx);
    classification = like_ratio > gamma;
    %need to calculate true positives
    % first we need to get the true positives these are all places where
    true_positive_indices = labels == 1;
    true_positive_rate(gamma_idx) = sum(classification(true_positive_indices)==1)/sum(true_positive_indices);
    false_positive_rate(gamma_idx) = sum(classification(~true_positive_indices)==1)/sum(~true_positive_indices);
    false_negative_rate(gamma_idx) = sum(classification(true_positive_indices)==0)/sum(true_positive_indices);
    true_negative_rate(gamma_idx) = sum(classification(~true_positive_indices)==0)/sum(~true_positive_indices);
    
    %p(error) = p(L1|Lo)P(L0) + P(L0|L1)P(L1)
    error_rate(gamma_idx) = false_positive_rate(gamma_idx) * prior_0 + false_negative_rate(gamma_idx) * prior_1;
    
    if error_rate(gamma_idx) < optimal_error_rate
        optimal_error_rate = error_rate(gamma_idx);
        optimal_gamma = gamma_values(gamma_idx);
        optimal_true_pos_rate = true_positive_rate(gamma_idx);
        optimal_false_pos_rate = false_positive_rate(gamma_idx);
        %sum(classification)
    end
end
%% analytical
analytical_gamma = prior_0/prior_1;%analytical ration
classification = like_ratio > analytical_gamma;%if greater than threshold then class 1
true_positive_indices = labels == 1;
true_positive_rate_analytical_gamma = sum(classification(true_positive_indices)==1)/sum(true_positive_indices);
false_positive_rate_analytical_gamma = sum(classification(~true_positive_indices)==1)/sum(~true_positive_indices);
false_negative_rate_analytical_gamma = sum(classification(true_positive_indices)==0)/sum(true_positive_indices);
true_negative_rate_analytical_gamma = sum(classification(~true_positive_indices)==0)/sum(~true_positive_indices);

%p(error) = p(L1|Lo)P(L0) + P(L0|L1)P(L1)
error_rate_analytical_gamma = false_positive_rate_analytical_gamma * prior_0 + false_negative_rate_analytical_gamma * prior_1;

plot(false_positive_rate,true_positive_rate)
hold on
scatter(optimal_false_pos_rate,optimal_true_pos_rate)
scatter(...
    false_positive_rate_analytical_gamma,...
    true_positive_rate_analytical_gamma)

hold off;
