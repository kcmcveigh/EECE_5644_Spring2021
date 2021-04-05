clear;
% Kieran McVeigh
% Optimal classifier is one that selects class with highest posterior for
% each point

mean_one_vector = [-1; 1; -1];
covariance_one_matrix =[1 0 0; 0 1.5 0.4; 0 0.4 .6];

mean_two_vector = [1; 1; 1];
covariance_two_matrix =[1.5 0.4 0; 0.4 1 0; 0 0 .7];

mean_three_vector = [1; 1; -1];
covariance_three_matrix =[.6 0 0.25; 0 1 0; 0.25 0 1];

mean_four_vector = [-1; -1; -1];
covariance_four_matrix =[.5 0 0.25; 0 1 0; 0.25 0 1];

load('Question1_data/100000_question_data.txt')

all_data = X100000_question_data;

posterior_1 = mvnpdf(all_data(:,1:3),mean_one_vector', covariance_one_matrix);
posterior_2 = mvnpdf(all_data(:,1:3),mean_two_vector', covariance_two_matrix);
posterior_3 = mvnpdf(all_data(:,1:3),mean_three_vector', covariance_three_matrix);
posterior_4 = mvnpdf(all_data(:,1:3),mean_four_vector', covariance_four_matrix);

posteriors = [posterior_1 posterior_2 posterior_3 posterior_4];
[max_posterior, classifier_labels] = max(posteriors,[],2);

%% build confusion
n_classes = 4;
confusion_matrix = ones(n_classes);
for true_label=1:n_classes
    
    true_labels_idx = all_data(:,4) == true_label;
    
    class_true_labels = all_data(true_labels_idx,4);
    class_classifier_labels = classifier_labels(true_labels_idx);
    
    n_true_class_members = length(class_true_labels);
    
    for classifier_label=1:n_classes
        accuracy=sum(class_classifier_labels==classifier_label)/n_true_class_members;
        confusion_matrix(classifier_label,true_label) = accuracy;
    end
end

p_error = 1-mean(classifier_labels == all_data(:,4))
