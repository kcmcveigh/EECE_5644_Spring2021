%Question2_PartA
% Kieran McVeigh 2/21/21
clear;
%% load data and declare distributions
load('all_data_question_two.txt')

%mean vectors and covariance used to generate the data in
%Question2_GenerateData.m
mean_1_vector = [1 0 0];
mean_2_vector =[1 1 1];
mean_3a_vector = [0 1 0];%3a
mean_3b_vector = [0 0 1];%3b

covariance = [.75 0 0; 0 .75 0; 0 0 .75];
prior_1 = .3; prior_2 = .3; prior_3=.4;

%% loss matrix swap in and out 
loss_matrix = [0 1 10 ; 1 0 10 ; 1 1 0 ];
loss_matrix = [0 1 100 ; 1 0 100 ; 1 1 0 ];

%% get approx posteriors
posterior_1 = mvnpdf(all_data_question_two(:,1:3),mean_1_vector,covariance)*prior_1;
posterior_2 = mvnpdf(all_data_question_two(:,1:3),mean_2_vector,covariance)*prior_2;
posterior_3a = mvnpdf(all_data_question_two(:,1:3),mean_3a_vector,covariance)*.5;
posterior_3b = mvnpdf(all_data_question_two(:,1:3),mean_3b_vector,covariance)*.5;
posterior_3 = (posterior_3a + posterior_3b) * prior_3;%get posterior for class across two distributions

posteriors = [posterior_1 posterior_2 posterior_3];

%% classify
% calculate_loss by doing the following: multiply the posterior times the
% loss then sum it
n_posteriors = length(posteriors);
min_risk=ones(n_posteriors,1);%initialize min risk vector
min_risk_class=ones(n_posteriors,1);%initialize decision with min risk vector
for obs_idx=1:length(posteriors)
    class_decision_loss=loss_matrix*posteriors(obs_idx,1:3)';%calculate risk for each class
    %get class with minimum risk
    [min_risk(obs_idx), min_risk_class(obs_idx)] = min(class_decision_loss);
end

classifier_labels = min_risk_class;%just for keeping names consistent
%% build confusion
confusion_matrix = ones(3);
true_classes=all_data_question_two(:,4);
for true_label=1:3
    
    true_labels_idx = true_classes == true_label;
    class_true_labels = true_classes(true_labels_idx);%get labels for class data
    class_classifier_labels = classifier_labels(true_labels_idx);% get classifier labels
    
    n_true_class_members = length(class_true_labels);%total observations of that class
    
    for classifier_label=1:3
        %get the true labels
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

%% minimal risk
% minimal risk for each observation times (sum of posteriors)
% sum of approx posteriors p(x) cause we say p(x) = integral (p(x|l)p(l)) wrt l
overall_risk = sum(min_risk.*sum(posteriors,2));


