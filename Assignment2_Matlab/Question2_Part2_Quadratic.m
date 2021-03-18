clear;
%Question2_Part2
% inspiration taken from HW examples from Summer2_2020 take home examples
% from matlab
%% set up data
data_file_string = 'question2_2000.txt';
load(strcat('Question2Data/',data_file_string));
load('Question2Data/question2_10000.txt');
train_data =question2_2000(:,1:2);
% create quadratic term
train_data(:,3:4) = train_data(:,1:2).*train_data(:,1:2);
train_data(:,5) = train_data(:,1).*train_data(:,2);
train_data(:,end+1)=1;
train_labels =  question2_2000(:,3);
weights_init = [0 0 0 0 0 0];

test_data = question2_10000(:,1:2);
test_data(:,3:4) = test_data(:,1:2).*test_data(:,1:2);
test_data(:,5) = test_data(:,1).*test_data(:,2);
test_data(:,end+1)=1;
test_labels = question2_10000(:,3);

%% train LogisticCost(data,labels,weights)
trained_weights = fminsearch(...
    @(weights)LogisticCost(train_data,train_labels,weights),...
    weights_init...
);

%% classify 
y_hat=1./(1+exp(-test_data*trained_weights'));
classification = y_hat>.5;
correct = classification == test_labels;

%% Calc error rate 
prior_L0 = .65;
prior_L1 = .35;

false_positives =classification(classification==1 & ~correct);
false_positive_rate = sum(false_positives)/sum(test_labels==0);

false_negatives = classification(classification==0 & test_labels==1);
false_negative_rate =length(false_negatives)/sum(test_labels==1);
error_rate= false_positive_rate * prior_L0 ...
    + false_negative_rate * prior_L1;


class_markers=strings(length(test_labels),1);

colors =zeros(length(test_labels),3);
colors(correct,2)=1;
colors(~correct,1)=1;



hold on
scatter(test_data(test_labels==0,1),test_data(test_labels==0,2),...
    20,colors(test_labels==0,:),'o','filled');
scatter(test_data(test_labels==1,1),test_data(test_labels==1,2),...
    40,colors(test_labels==1,:),'+');
hold off
title(data_file_string);

function [cost] =LogisticCost(data,labels,weights)
    y_hat=1./(1+exp(-data*weights'));
    cost = -sum(labels.*log(y_hat)+(1-labels).*log(1-y_hat))/length(y_hat);
end

