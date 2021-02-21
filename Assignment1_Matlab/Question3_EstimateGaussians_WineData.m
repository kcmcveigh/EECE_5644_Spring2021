% Kieran McVeigh 2/21/21
% estimate gaussians for wine dataset
wine_data =importdata('winequality-white.csv');
feature_data = wine_data.data(:,1:11);
labels = wine_data.data(:,12);
feature_cov_matrix = cov(feature_data);

alpha=.5;

%% initialize feature means, covariance and priors
different_labels = unique(wine_data.data(:,12));
feature_means = zeros(10,11);
feature_cov = zeros(10,11,11);
label_counts = zeros(10,2);

%% estimate feature means covs and priors
for label_idx=1:10 %for each class estimate paramaters
    label=label_idx;
    class_feature_data = feature_data(labels==label,:);%get class data
    label_counts(label,1) = sum(labels==label);%get n instances of each class
    label_counts(label,2) = label_counts(label,1)/length(feature_data);%get p(class)
    feature_means(label,:) = mean(class_feature_data,1);%estimate mean vector for class
    cov_sample =cov(class_feature_data);%calc class sample cov
    if ~isnan(cov_sample(1,1))% regularize
        reg = .01 * trace(cov_sample)/rank(cov_sample);
        cov_reg = cov_sample + eye(11)*.01;%*reg;
    else
        cov_reg = nan(11,11);
    end
    feature_cov(label,:,:) = cov_reg;
end

%save data
wine_feature_means = feature_means;
wine_cov =feature_cov;
wine_priors = label_counts(:,2);

writematrix(wine_feature_means);
writematrix(wine_cov);
writematrix(wine_priors);