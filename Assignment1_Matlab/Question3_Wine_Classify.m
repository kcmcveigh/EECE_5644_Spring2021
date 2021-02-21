% Kieran McVeigh 2/21/21
clear
%% classify wine data based on estimated gaussian parameters
load('wine_cov.txt')
load('wine_priors.txt')
load('wine_feature_means.txt')

wine_data =importdata('winequality-white.csv');
feature_data = wine_data.data(:,1:11);
labels = wine_data.data(:,12);

%create posterior matrix - for each obs by 10 (classes)
posteriors = zeros(length(labels),10);% initialize matrix
for class_idx=1:10
    if(wine_priors(class_idx)~=0)%only do this calc if there are instances of this class
        class_cov_matrix = reshape(wine_cov(class_idx,:,:),11,11);
        class_mean = wine_feature_means(class_idx,:);
        likelihood = mvnpdf(feature_data,class_mean,class_cov_matrix);
        class_prior = wine_priors(class_idx);
        posteriors(:,class_idx) = likelihood *class_prior;
    end
end

[max_post, classifier_labels] = max(posteriors,[],2);

confusion_matrix = ones(10,10);
for true_label=1:10
    
    true_labels_idx = labels == true_label;
    class_true_labels = labels(true_labels_idx);
    class_classifier_labels = classifier_labels(true_labels_idx);
    
    n_true_class_members = length(class_true_labels);
    for classifier_label=1:10
        %get the true labels
        accuracy=sum(class_classifier_labels==classifier_label)/n_true_class_members;
        if ~isnan(accuracy)
            confusion_matrix(classifier_label,true_label) = accuracy;
        else
            confusion_matrix(classifier_label,true_label) =0;
        end
    end
    pcaed_feature_data =feature_data*pca(feature_data,"NumComponents",3);
    scatter3(pcaed_feature_data(true_labels_idx,1),...
        pcaed_feature_data(true_labels_idx,2),...
        pcaed_feature_data(true_labels_idx,3),...
        100, 'filled')
        %
    hold on
end

hold off
p_error = 0;
for true_label=1:10
    class_prior = wine_priors(true_label);
    for class_label=1:10
        if(true_label~=class_label)
            conditional_error =confusion_matrix(class_label,true_label);
            p_error = p_error + class_prior*conditional_error;
        end
    end
end

p_error
confusion_matrix

