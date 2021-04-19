% Kieran McVeigh
% GMM two components
clear;

filenames{1,1} = '3096_color.jpg';
%filenames{1,2} = '42049_color.jpg';
cluster_list = [1,2,3,4,5,6,7,8,9,10];
cluster_list = [15,20,25,30];
for imageCounter = 1:1%size(filenames,2)
    %%data set up
    imdata = imread(filenames{1,imageCounter}); 
    n_plot_cols=length(cluster_list)+1;
    figure(1), subplot(size(filenames,2),n_plot_cols,(imageCounter-1)*n_plot_cols+1), imshow(imdata);

    [R,C,D] = size(imdata); N = R*C; imdata = double(imdata);
    rowIndices = [1:R]'*ones(1,C); colIndices = ones(R,1)*[1:C];
    features = [rowIndices(:)';colIndices(:)']; % initialize with row and column indices
    for d = 1:D
        imdatad = imdata(:,:,d); % pick one color at a time
        features = [features;imdatad(:)'];
    end
    minf = min(features,[],2); maxf = max(features,[],2);
    ranges = maxf-minf;
    x = diag(ranges.^(-1))*(features-repmat(minf,1,N));
    x=x';
    n_samples = length(x);
    n_features=5;
    for n_clusters_idx=1:length(cluster_list)
         n_clusters = cluster_list(n_clusters_idx)
        %% intialize gmm
        q_prev = 0;
        %n_clusters =2;

        priors = ones(n_clusters,1); priors = (priors./sum(priors))';
        means=x(randi(length(x),n_clusters,1),:);
        n_features=5;
        for sig_idx=1:n_clusters
            sigma(:,:,sig_idx)=eye(n_features);
        end


        %% estimate gmm
        converged = 0;
        while ~converged
        %% Expectation
            [q,posteriors] = ExpectationStep(...
                n_clusters,...%clusters
                x,...
                means,...
                sigma,...
                priors);
            %% maximization
            [priors,means,sigma] = MaximizationStep(n_clusters,x,means,sigma,posteriors);
            converged = abs(q-q_prev) < .01;
            q_prev = q;
            
            
        end
        q
        
        %% my code
        [~,labels] = max(posteriors,[],2);
        labelImage = reshape(labels,R,C);
         figure(1), subplot(size(filenames,2),length(cluster_list)+1,(imageCounter-1)*(length(cluster_list)+1)+1+n_clusters_idx), imshow(uint8(labelImage*255/n_clusters));
         n = (n_clusters-1) + n_features*n_clusters + n_clusters*(n_features*(n_features+1)/2);
         complexity_penalty = n * log(n_features*n_samples);

         [log_like] = LogLikelihood(means,sigma,x,priors);

         bic(n_clusters_idx)=-2*log_like+complexity_penalty;
         
         
    end
end

function [log_like] = LogLikelihood(means,sigmas,x,priors)
    n_clusters = size(means,1);
    for cluster_idx=1:n_clusters
        sigmas(:,:,cluster_idx)=sigmas(:,:,cluster_idx)+eye(5)*.001;
        likelihood(:,cluster_idx) = mvnpdf(x,means(cluster_idx,:),sigmas(:,:,cluster_idx))*priors(cluster_idx);
    end
    log_like=sum(log(sum(likelihood,2)));
end
