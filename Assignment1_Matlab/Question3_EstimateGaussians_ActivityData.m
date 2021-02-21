% Kieran McVeigh 2/21/21
%estimate class conditional gaussian parameters
clear

%% load data
% Paths have been changes to data for code storage reasons

activity_train_features =importdata('UCI HAR Dataset/train/X_train.txt');
activity_train_labels = importdata('UCI HAR Dataset/train/Y_train.txt');

activity_test_features =importdata('UCI HAR Dataset/test/X_test.txt');
activity_test_labels = importdata('UCI HAR Dataset/test/Y_test.txt');

activity_labels = [activity_train_labels;activity_test_labels];
activity_features = [activity_train_features;activity_test_features];

%% initialize data set info
n_labels = 6;
n_features = 561;
alpha=.5;

feature_means = zeros(n_labels,n_features);
feature_cov = zeros(n_labels,n_features,n_features);
label_counts = zeros(n_labels,2);

feature_data = activity_features;
labels = activity_labels;

%% estimate parameters
for label_idx=1:n_labels
    label=label_idx;
    class_feature_data = feature_data(labels==label,:);
    
    label_counts(label,1) = sum(labels==label);
    label_counts(label,2) = label_counts(label,1)/length(feature_data);
    
    feature_means(label,:) = mean(class_feature_data,1);
    
    cov_sample =cov(class_feature_data);
    if ~isnan(cov_sample(1,1))
        reg = .01 * trace(cov_sample)/rank(cov_sample);
        cov_reg = cov_sample + eye(n_features)*reg;
    else
        cov_reg = nan(n_features,n_features);
    end
    feature_cov(label,:,:) = cov_reg;
end
%% save estimates 
activity_feature_means = feature_means;
activity_cov =feature_cov;
activity_priors = label_counts(:,2);

writematrix(activity_feature_means);
writematrix(activity_cov);
writematrix(activity_priors);