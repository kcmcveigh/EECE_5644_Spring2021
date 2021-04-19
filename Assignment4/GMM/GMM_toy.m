% Kieran McVeigh
% GMM - EM 
clear
%% toy example data
load('samples_X.txt')
% ok so what are the steps - initialize
% calculate Expectation over z - or class assignemtn
% update parameters - repeat
n_clusters = 3;
%n_samples = length(all_data);
n_samples=250
%samples_X = all_data(:,1:3);
labels = ones(n_samples,1);
for n_clusters=1:8
    n_features=3;
%% initialize
    priors = ones(n_clusters,1); priors = (priors./sum(priors))';
    initial_means_idx = randi(n_samples,n_clusters,1);
    means = samples_X(initial_means_idx,1:n_features);
    %sigma=eye(3);
    for sig_idx=1:n_clusters
        sigma(:,:,sig_idx)=eye(n_features);
    end
    q_prev = 0;
    converged=0;

    while ~converged
    %% Expectation
    %     for cluster_idx=1:n_clusters
    %         likelihoods(:,cluster_idx) = mvnpdf(samples_X,...
    %             means(cluster_idx,:),...
    %             sigma(:,:,cluster_idx));
    %     end
    %     % estimate join and posterior
    %     joint = likelihoods .* priors;
    %     posterior =joint./sum(joint,2);% soft cluster
    %     log_joint = log(joint);
    %     q = sum(sum(posterior.*log_joint))

    [q,posteriors] = ExpectationStep(...
        n_clusters,...
        samples_X,...
        means,...
        sigma,...
        priors);

        %% maximization
    [priors,means,sigma] = MaximizationStep(n_clusters,samples_X,means,sigma,posteriors);
        [~,labels] = max(posteriors,[],2);
        converged = abs(q-q_prev) < .01;
        q_prev = q
        
        
    end
    n = (n_clusters-1) + n_features*n_clusters + n_clusters*(n_features*(n_features+1)/2);
    complexity_penalty = n * log(n_features*n_samples);

    [log_like] = LogLikelihood(means,sigma,samples_X,priors);

    bic(n_clusters)=-2*log_like+complexity_penalty

     %% Validation based on BICforGMM.m from class files
    options = statset('MaxIter',1000); % Specify max allowed number of iterations for EM
        % Run EM 'Replicates' many times and pickt the best solution
        % This is a brute force attempt to catch the globak maximum of
        % log-likelihood function during EM based optimization
    gm{n_clusters} = fitgmdist(samples_X,n_clusters,'Replicates',2,'Options',options);
    neg2logLikelihood(1,n_clusters) = -2*sum(log(pdf(gm{n_clusters},samples_X)));
    BIC(1,n_clusters) = neg2logLikelihood(1,n_clusters)+complexity_penalty;
end

function [log_like] = LogLikelihood(means,sigmas,x,priors)
    n_clusters = size(means,1);
    for cluster_idx=1:n_clusters
        likelihood(:,cluster_idx) = mvnpdf(x,means(cluster_idx,:),sigmas(:,:,cluster_idx))*priors(cluster_idx);
    end
    log_like=sum(log(sum(likelihood,2)));
end



