% Kieran McVeigh 2/21/21

% load data % first 4 columns are features and 5th column is labels
all_data = importdata('all_data.txt');

%% setup from HW
mean_zero_vector = [-1; 1; -1; 1];
covariance_zero_matrix =[2 -.5 .3 0; -.5 1 -.5 0; 0.3 -0.5 1 0; 0 0 0 2];

mean_one_vector = [1; 1; 1; 1];
covariance_one_matrix =[1 .3 -.2 0; .3 2 .3 0; -0.2 0.3 1 0; 0 0 0 3];
labels = all_data(:,5);
prior_L0 = .7;
prior_L1 = .3;

%% calculate likelhood ration
like_zero = mvnpdf(all_data(:,1:4),mean_zero_vector',covariance_zero_matrix);
like_one = mvnpdf(all_data(:,1:4),mean_one_vector',covariance_one_matrix);

like_ratio = like_one(:)./like_zero(:);


%% empirical gamma
gamma_values = linspace(0,1000,100000);
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
    error_rate(gamma_idx) = false_positive_rate(gamma_idx) * prior_L0 + false_negative_rate(gamma_idx) * prior_L1;
    
    if error_rate(gamma_idx) < optimal_error_rate
        optimal_error_rate = error_rate(gamma_idx);
        optimal_gamma = gamma_values(gamma_idx);
        optimal_true_pos_rate = true_positive_rate(gamma_idx);
        optimal_false_pos_rate = false_positive_rate(gamma_idx);
    end
end
%% analytical
analytical_gamma = .7/.3;%analytical ration
classification = like_ratio > analytical_gamma;%if greater than threshold then class 1
true_positive_indices = labels == 1;
true_positive_rate_analytical_gamma = sum(classification(true_positive_indices)==1)/sum(true_positive_indices);
false_positive_rate_analytical_gamma = sum(classification(~true_positive_indices)==1)/sum(~true_positive_indices);
false_negative_rate_analytical_gamma = sum(classification(true_positive_indices)==0)/sum(true_positive_indices);
true_negative_rate_analytical_gamma = sum(classification(~true_positive_indices)==0)/sum(~true_positive_indices);

%p(error) = p(L1|Lo)P(L0) + P(L0|L1)P(L1)
error_rate_analytical_gamma = false_positive_rate_analytical_gamma * prior_L0 + false_negative_rate_analytical_gamma * prior_L1;

plot(false_positive_rate,true_positive_rate)
hold on
scatter(optimal_false_pos_rate,optimal_true_pos_rate)
scatter(...
    false_positive_rate_analytical_gamma,...
    true_positive_rate_analytical_gamma)

hold off;



