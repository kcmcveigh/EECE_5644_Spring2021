% Kieran McVeigh 2/21/21
% classify activity data based on estimated class gaussian parameters
clear

load('activity_cov.txt')
load('activity_priors.txt')
load('activity_feature_means.txt')

activity_train_features =importdata('../../UCI HAR Dataset/train/X_train.txt');
activity_train_labels = importdata('../../UCI HAR Dataset/train/Y_train.txt');

activity_test_features =importdata('../../UCI HAR Dataset/test/X_test.txt');
activity_test_labels = importdata('../../UCI HAR Dataset/test/Y_test.txt');

activity_labels = [activity_train_labels;activity_test_labels];
activity_features = [activity_train_features;activity_test_features];

n_labels = 6;
n_features = 561;

%% initialize general variables
labels = activity_labels;
feature_data = activity_features;

priors = activity_priors;
feature_cov = activity_cov;
feature_means = activity_feature_means;

%% calc posteriors
posteriors = zeros(length(labels),n_labels);
for class_idx=1:n_labels
    if(priors(class_idx)~=0)
        class_cov_matrix = reshape(feature_cov(class_idx,:,:),n_features,n_features);
        class_mean = feature_means(class_idx,:);
        class_like = mvnpdf(feature_data,class_mean,class_cov_matrix);
        class_prior = activity_priors(class_idx);
        posteriors(:,class_idx) = class_like*class_prior;
    end
end

%get Maximum posterior class - use as class label
[max_post, classifier_labels] = max(posteriors,[],2);

%% calculate confusion matrix
confusion_matrix = ones(n_labels);
for true_label=1:n_labels%columns
    
    %get true class labels
    true_labels_idx = labels == true_label;
    class_true_labels = labels(true_labels_idx);
    class_classifier_labels = classifier_labels(true_labels_idx);
    
    n_true_class_members = length(class_true_labels);
    for classifier_label=1:n_labels%rows
        %calc percent of instances classified as each class for a true
        %class label
        accuracy=sum(class_classifier_labels==classifier_label)/n_true_class_members;
        if ~isnan(accuracy)
            confusion_matrix(classifier_label,true_label) = accuracy;
        else
            confusion_matrix(classifier_label,true_label) =0;
        end
    end
    
    %plot each class in 3 space
    pcaed_feature_data =feature_data*pca(feature_data,"NumComponents",3);
    scatter3(pcaed_feature_data(true_labels_idx,1),...
        pcaed_feature_data(true_labels_idx,2),...
        pcaed_feature_data(true_labels_idx,3),...
        20)
    hold on
end
legend('1','2','3','4','5','6')
hold off 

% calc p(error)
p_error = 0;
for true_label=1:n_labels
    class_prior = priors(true_label);
    for class_label=1:n_labels
        if(true_label~=class_label)
            conditional_error =confusion_matrix(class_label,true_label);
            p_error = p_error + class_prior*conditional_error;
        end
    end
end

