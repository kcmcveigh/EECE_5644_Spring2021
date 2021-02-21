%Question2_PartA
% Kieran McVeigh 2/21/21
clear
%% load data and declare distributions
load('all_data_question_two.txt')

%mean vectors and covariance used to generate the data in
%Question2_GenerateData.m
mean_1_vector = [1 0 0];
mean_2_vector =[1 1 1];
mean_3a_vector = [0 1 0];%3a
mean_3b_vector = [0 0 1];%3b

covariance = [.75 0 0; 0 .75 0; 0 0 .75];

%% get approx posteriors
prior_1 = .3; prior_2 = .3; prior_3=.4;

%% get approx posteriors
posterior_1 = mvnpdf(all_data_question_two(:,1:3),mean_1_vector,covariance)*prior_1;
posterior_2 = mvnpdf(all_data_question_two(:,1:3),mean_2_vector,covariance)*prior_2;

posterior_3a = mvnpdf(all_data_question_two(:,1:3),mean_3a_vector,covariance)*.5;
posterior_3b = mvnpdf(all_data_question_two(:,1:3),mean_3b_vector,covariance)*.5;
posterior_3 = (posterior_3a + posterior_3b) * prior_3;

posteriors = [posterior_1 posterior_2 posterior_3];

%% classify
[max_posterior, max_posterior_idx] = max(posteriors,[],2);
%convert all 4s to 3s since both posteriors for 3a/b are classified as 3
classifier_labels = max_posterior_idx;

%% build confusion
confusion_matrix = ones(3);
for true_label=1:3
    
    true_labels_idx = all_data_question_two(:,4) == true_label;
    
    class_true_labels = all_data_question_two(true_labels_idx,4);
    class_classifier_labels = classifier_labels(true_labels_idx);
    
    n_true_class_members = length(class_true_labels);
    
    for classifier_label=1:3
        accuracy=sum(class_classifier_labels==classifier_label)/n_true_class_members;
        confusion_matrix(classifier_label,true_label) = accuracy;
    end
end

%% visualize
true_class_labels = all_data_question_two(:,4);
correct_classification = classifier_labels == true_class_labels;

label_markers = ['o','x','^'];
colors = ['r','g'];
for true_class=1:3
    class_data_idx = true_class_labels==true_class;
    for correct=1:2
        class_data = all_data_question_two(class_data_idx & correct_classification==(correct-1),:);
        scatter3(...
            class_data(:,1),...
            class_data(:,2),...
            class_data(:,3),...
            10*ones(length(class_data),1),...
            colors(correct),...
            label_markers(true_class))
        hold on
    end
end





