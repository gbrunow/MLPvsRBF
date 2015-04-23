clear all;
clc;
delete(findall(0,'Type','figure'));
bdclose('all');

% load('training.mat');
% load('validation.mat');
% 
% input = tr_features;
% desired_out = tr_labels;

data = load('dataset.csv');
data = data(randperm(size(data,1)),:);

dataset_size = size(data(:,end),1);
n_out = zeros([dataset_size 2]);
n_out(find(data(:,end) == 1), 1) = 1;
n_out(find(data(:,end) == 2), 2) = 1;

data = [data(:,1:end-1) n_out];

folds = 4;
dataset_size = floor(dataset_size/folds)*folds;
data = data(1:dataset_size, :);

%neural network settings
learning_rate = 0.0001;
alpha = 0.7;
k=1;
for hidden_neurons = 10:10:180
    clc;
    delete(findall(0,'Type','figure'));
    bdclose('all');
    
    if k > 1
        figure(2);
        plot(neurons, percentage_vec);
        drawnow;
    end

    fold_end = dataset_size/folds;
    input = zeros(size([fold_end:dataset_size], 2), 2);
    input = [input ones(size(input,1),1) ]; 

    %input layer weights
    w1 = 0.5*(1-2*rand(size(input,2),hidden_neurons-1)); 

    %hidden layer weights
    w2 = 0.5*(1-2*rand(hidden_neurons,2));

    %variables initialization        
    epoch = 0;
    error_trace = [];
    error = inf;
    dw1_previous = zeros(size(w1)); 
    dw2_previous = zeros(size(w2)); 
    figure;
    tic
    right = 0;
    while error > 60
        epoch_error = 0;
        for current_fold = 1:folds

            fold_beginning = dataset_size*(current_fold - 1)/folds;
            fold_end = dataset_size*current_fold/folds;

            input(:,1:2) = data([1:fold_beginning fold_end:dataset_size],1:2);
            desired_out = data([1:fold_beginning fold_end:dataset_size],3:4);

            if(fold_beginning < 1)
                fold_beginning = 1;
            end

            val_features = data(fold_beginning:fold_end,1:2);
            val_labels = data(fold_beginning:fold_end,3:4);
            hidden = [1./(1+exp(-input * w1)) ones(size(input,1),1)];
            output = 1./(1+exp(-hidden * w2));
            output_error = desired_out - output;

            epoch_error = epoch_error + trace(output_error'*output_error);
            error_trace = [error_trace error];

            deltas_out = output_error .* output .* (1-output);
            deltas_hid = deltas_out*w2' .* hidden .* (1-hidden);
            deltas_hid(:,size(deltas_hid,2)) = []; 

            dw1 = learning_rate * input' * deltas_hid + alpha * dw1_previous;   
            dw2 = learning_rate * hidden' * deltas_out + alpha * dw2_previous;
            w1 = w1 + dw1; 
            w2 = w2 + dw2;
            dw1_previous = dw1;
            dw2_previous = dw2;
        end
        error = epoch_error/folds;
        epoch = epoch + 1;
        if rem(epoch,50)==0

            right = 0;
            for i = 1:size(val_features,1)
                weighted_input = [val_features(i,:) 1]*w1;
                weighted_hidden = [1./(1+exp(-weighted_input)) 1]*w2;
                output = 1./(1+exp( - weighted_hidden));
                output(output > 0.5) = 1;
                output(output < 0.5) = 0;
                if output == val_labels(i,:)
                   right = right + 1;
                end
            end
            percentage(epoch/50) = right/size(val_labels,1);
            subplot(2,1,1);
            plot(error_trace);
            xlabel('Epochs');
            ylabel('Error');
            subplot(2,1,2);
            plot(percentage, 'r');
            xlabel('Epochs');
            ylabel('Hits');
            drawnow;
            clc;
            toc
        end
    end
    toc
    percentage = 100*right/size(val_labels,1);
    percentage_vec(k) = percentage;
    neurons(k) = hidden_neurons;
    figure(2);
    plot(neurons, percentage_vec);
    drawnow;
    k = k+1;
    disp([' Correctness ' num2str(percentage) '%']);
end