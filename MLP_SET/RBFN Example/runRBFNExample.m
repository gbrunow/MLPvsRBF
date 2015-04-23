% ======== runRBFNExample ========
% This script trains an RBF Network on an example dataset, and plots the
% resulting score function and decision boundary.
% 
% There are three main steps to the training process:
%   1. Prototype selection through k-means clustering.
%   2. Calculation of beta coefficient (which controls the width of the 
%      RBF neuron activation function) for each RBF neuron.
%   3. Training of output weights for each category using gradient descent.
%
% Once the RBFN has been trained this script performs the following:
%   1. Generates a contour plot showing the output of the category 1 output
%      node.
%   2. Shows the original dataset with the placements of the protoypes and
%      an approximation of the decision boundary between the two classes.
%   3. Evaluates the RBFN's accuracy on the training set.

% $Author: ChrisMcCormick $    $Date: 2014/08/18 22:00:00 $    $Revision: 1.3 $

% Clear all existing variables from the workspace.
clear all;
clc;

% Add the subdirectories to the path.
addpath('kMeans');
addpath('RBFN');

% Load the data set. 
% This loads two variables, X and y.
%   X - The dataset, 1 sample per row.
%   y - The corresponding label (category 1 or 2).
% The data is randomly sorted and grouped by category.

load('training.mat');
load('validation.mat');

X = tr_features;

for i=1:size(tr_labels,1)
    if tr_labels(i,1) == 1
        y(i,:) = 1;
    elseif tr_labels(i,2) == 1
        y(i,:) = 2;
    elseif tr_labels(i,3) == 1
        y(i,:) = 3;
    elseif tr_labels(i,4) == 1
        y(i,:) = 4;
    elseif tr_labels(i,5) == 1
        y(i,:) = 5;
    elseif tr_labels(i,6) == 1
        y(i,:) = 6;
    end        
end

% Set 'm' to the number of data points.
m = size(X, 1);

% ===================================
%     Train RBF Network
% ===================================

k = 1;
for c = 10:10:180
    disp('Training the RBFN...');

    % Train the RBFN using 10 centers per category.
    [Centers, betas, Theta] = trainRBFN(X, y, c, true);

    % ================================
    %         Contour Plots
    % ================================

    disp('Evaluating RBFN over input space...');

    % Evaluate the RBFN over the grid.
    % For each row of the grid...

    for i=1:size(val_labels,1)
        if val_labels(i,1) == 1
            output(i,:) = 1;
        elseif val_labels(i,2) == 1
            output(i,:) = 2;
        elseif val_labels(i,3) == 1
            output(i,:) = 3;
        elseif val_labels(i,4) == 1
            output(i,:) = 4;
        elseif val_labels(i,5) == 1
            output(i,:) = 5;
        elseif val_labels(i,6) == 1
            output(i,:) = 6;
        end        
    end

    input = val_features;
    right = 0;
    for (i = 1 : size(input,1))

            % Compute the category scores.
            scores = evaluateRBFN(Centers, betas, Theta, input(i,:));
            [a, b] = max(scores);
            if(b == output(i))
                right = right + 1;
            end
    end
    correctness(k) = right/size(input,1);
    cnt(k) = c;
    plot(cnt, correctness);
    drawnow;
    k = k + 1;
end