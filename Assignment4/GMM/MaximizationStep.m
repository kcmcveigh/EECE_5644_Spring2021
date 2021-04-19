function [priors,means,sigma] = MaximizationStep(n_clusters,samples_X,means,sigma,posteriors)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    n_samples=length(samples_X);
    priors = sum(posteriors)/n_samples;
    for cluster_idx=1:n_clusters
        class_posterior = posteriors(:,cluster_idx);
        weighted_count =n_samples*priors(cluster_idx);
        means(cluster_idx,:) = class_posterior'*samples_X/weighted_count;
        mean_diff = samples_X - means(cluster_idx,:);
        sigma(:,:,cluster_idx) = class_posterior'.*mean_diff'*mean_diff/weighted_count;
    end
end

