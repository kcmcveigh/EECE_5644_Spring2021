% Kieran McVeigh
% GMM two components
clear;

filenames{1,1} = '3096_color.jpg';
filenames{1,2} = '42049_color.jpg';
for imageCounter = 1:2%size(filenames,2)
    %%data set up
    imdata = imread(filenames{1,imageCounter}); 
    figure(1), subplot(size(filenames,2),1+1,(imageCounter-1)*(1+1)+1), imshow(imdata);

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
    
    %% intialize gmm
    q_prev = 0;
    n_clusters =2;
    
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
        q_prev = q
    end
    [~,labels] = max(posteriors,[],2);
    labelImage = reshape(labels,R,C);
     figure(1), subplot(size(filenames,2),1+1,(imageCounter-1)*(1+1)+1+1), imshow(uint8(labelImage*255/2));
    
end