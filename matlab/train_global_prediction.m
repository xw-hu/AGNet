%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Global Prediction Network Train
% Copyright (c) The Chinese University of Hong Kong
% @incollection{hu2017agnet,
%  title={AGNet: Attention-Guided Network for Surgical Tool Presence Detection},
%  author={Hu, Xiaowei and Yu, Lequan and Chen, Hao and Qin, Jing and Heng, Pheng-Ann},
%  booktitle={Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support},
%  pages={186--194},
%  year={2017},
%  publisher={Springer}
% }
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
caffe.reset_all();

caffe.set_mode_gpu();
caffe.set_device(0);

%configeration
max_iter = 10000;
batch_size = 16; 
do_mirror = true;
image_size = [224,224];
val_interval = 100000;
val_iter = 1000;
snapshot_interval = 3000;
average_loss_num = 50;

%caffe initilization
solver_def_file = './models/global_predcition/solver.prototxt';
net_file = './models/ResNet-101-model.caffemodel';

caffe_solver = caffe.Solver(solver_def_file);
caffe_solver.net.copy_from(net_file);

% init log
train_dir = fullfile(pwd, 'output/');
mkdir_if_missing(train_dir);
caffe_log_file_base = fullfile(train_dir, 'caffe_log');

timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
mkdir_if_missing(fullfile(train_dir, 'log'));
log_file = fullfile(train_dir, 'log', ['train_', timestamp, '.txt']);
diary(log_file);

%train parameters
iter_ = caffe_solver.iter();

%generate data
train_data = importdata('./data/tool_detection_train.txt');
test_data = importdata('./data/tool_detection_test.txt');
label = train_data.data;
image = train_data.textdata;
test_label = test_data.data;
test_image = test_data.textdata;

num_image = length(image);

% random order
choose = randperm(num_image)';
label = label(choose,:);
image = image(choose,:);
count = 0;

loss = zeros(max_iter,1);
loss_avg = zeros(floor(max_iter/100),1);
loss_val = zeros(floor(max_iter/val_interval),1);

while (iter_ < max_iter)
    %get data & generate batch
    image_batch_data = zeros(image_size(1),image_size(2),3,batch_size);
    label_batch_data = zeros(1,1,7,batch_size);
    for im = 1:batch_size
        
        count = count+1; 
        image_data = imread(image{count});
        
        % image pre-processing
        image_data = im_alignment(image_data);
        
        image_data = image_transfer(image_data, do_mirror, image_size);
        
        image_batch_data(:,:,:,im) = image_data;
        label_batch_data(1,1,:,im) = label(count,:);
        
        if (count == num_image)
            fprintf('Restart to train the data!\n');
            count = 0;
            choose = randperm(num_image)';
            label = label(choose,:);
            image = image(choose,:);
        end
    end
    
    caffe_solver.net.blobs('data').reshape([image_size(1) image_size(2) 3 batch_size]);
    caffe_solver.net.blobs('label').reshape([1 1 7 batch_size]);
    caffe_solver.net.reshape();
     
    % one iter SGD update
    caffe_solver.net.blobs('data').set_data(image_batch_data);
    caffe_solver.net.blobs('label').set_data(label_batch_data);
    caffe_solver.step(1);
   
    loss(iter_+1) = caffe_solver.net.blobs('loss').get_data(); %%count from 0
    
    if ~mod((iter_+1), average_loss_num) % avgerage training loss
        loss_avg((iter_+1)/average_loss_num,1) = sum(loss((iter_-average_loss_num+2):(iter_+1)))/average_loss_num;
        fprintf('Iterations: %d, Training Loss: %f\n',iter_+1,loss_avg((iter_+1)/average_loss_num,1));
    end
    
    if ~mod((iter_+1), val_interval) %%%validation
        
        val_loss = 0;
        % random order
        chos = randperm(length(test_image))';
        test_label = test_label(chos,:);
        test_image = test_image(chos,:);
        
        test_count = 0;
        
        for ti=1:floor(val_iter/batch_size)
            for im = 1:batch_size
                
                test_count = test_count+1;
                image_data = imread(test_image{test_count});
                image_data = im_alignment(image_data);
                image_data = image_transfer(image_data, false, image_size);
        
                image_batch_data(:,:,:,im) = image_data;
                label_batch_data(1,1,:,im) = test_label(test_count,:);
        
                if (count == num_image)
                     fprintf('Have already tested on whole test dataset!\n');
                     ti = val_iter/batch_size;
                     break;
                end
            end
            caffe_solver.net.blobs('data').reshape([image_size(1) image_size(2) 3 batch_size]);
            caffe_solver.net.blobs('label').reshape([1 1 7 batch_size]);
            caffe_solver.net.reshape();
     
            % one iter SGD update
            output = caffe_solver.net.forward({image_batch_data, label_batch_data});
            val_loss = val_loss + output{1};
        end
        loss_val((iter_+1)/val_interval) = val_loss/floor(val_iter/batch_size);
        fprintf('Iterations: %d, Validation Loss: %f\n',(iter_+1),loss_val((iter_+1)/val_interval));
        
    end
    
    if ~mod((iter_+1), snapshot_interval) 
        mkdir_if_missing(fullfile(train_dir, 'global_prediction'));
        model_path = fullfile(train_dir, 'global_prediction', sprintf('iter_%d.caffemodel', iter_+1));
        caffe_solver.net.save(model_path);
        fprintf('Saved as %s\n', model_path);
    end
        
    iter_ = caffe_solver.iter();
end
 
% final snapshot
model_path = fullfile('./output','global_prediction.caffemodel');
caffe_solver.net.save(model_path);
fprintf('Saved as %s\n', model_path);

% plot the training line
x = zeros(length(loss_avg),1);
for x_i=1:length(loss_avg)
    x(x_i,1) = average_loss_num*x_i;
end
h(1) = plot(x,loss_avg,'-+b');

hold on;

% plot the validation line
x = zeros(length(loss_val),1);
for x_i=1:length(loss_val)
    x(x_i,1) = val_interval*x_i;
end
%h(2) = plot(x,loss_val,'--*r');

hold on;
grid on;
xlabel('Iteration');
ylabel('Loss');
legend(h,'training loss','validation loss');
print(gcf,'-dpng','./output/global_prediction/training_curve.png');