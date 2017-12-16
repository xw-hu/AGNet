%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local Prediction Network Test
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

patch_size = [224,224];

model='./models/local_prediction/deploy.prototxt';
weight='./output/local_prediction.caffemodel';

% global prediction network
get_box_model = './models/global_predcition/deploy.prototxt';
get_box_weight = './output/global_prediction.caffemodel';
net=caffe.Net(model,weight,'test');
get_box_net=caffe.Net(get_box_model,get_box_weight,'test');

ground_truth_files = {'./data/test_label/tool_video_11.txt', ...
    './data/test_label/tool_video_12.txt', ...
    './data/test_label/tool_video_13.txt', ...
    './data/test_label/tool_video_14.txt', ...
    './data/test_label/tool_video_15.txt'};

image_dir = ['./data/test_image/' 'tool_video_'];

patch_image_batch_data = zeros(patch_size(1),patch_size(2),3,7);

for i = 1:length(ground_truth_files)
    
    ground_truth_file = ground_truth_files{i};
    pred_file = [ground_truth_file(1:end-4) '_pred.txt'];
    [gt, toolNames] = ReadToolAnnotationFile(ground_truth_file);
    
    fileFolder = strcat(image_dir,int2str(i+10));  %test
    file_name = strcat('/tool_video_',int2str(i+10));
    
    full_name = strcat(strcat(fileFolder,file_name),'_');
    
    usedtime = 0; avgtime = 0;
    for j=1:size(gt,1)
        final_jpg = strcat(full_name,int2str(gt(j,1)));
        final_jpg = strcat(final_jpg,'.jpg');
        
        %for each test image
        mu = ones(1,1,3); mu(:,:,1:3) = [67 70 90];
        imgH = 224; imgW = 224; %224
        mu = repmat(mu,[imgW,imgH,1]);
        
        test_image = imread(final_jpg);
        test_image = im_alignment(test_image);
        
        test_image = imresize(test_image,[imgH imgW]);
        test_image = single(test_image(:,:,[3 2 1]));
        test_image = bsxfun(@minus,test_image,mu);
        test_image = permute(test_image, [2 1 3]);
        
        %use the global prediction network
        tic; outputs = get_box_net.forward({test_image}); pertime=toc;
        usedtime=usedtime+pertime; avgtime=usedtime/j;
        
        res = squeeze(outputs{1});
        cls_feat = get_box_net.blobs('cls_conv').get_data();
        
        for cls = 1:7
            
            %OTSU algorithm (find the local patch)
            feature_cls = (cls_feat(:,:,cls));
            level = multithresh(feature_cls);
            BW = imquantize(feature_cls,level);
            tmp_box = regionprops(BW,'BoundingBox');
            x1 = (ceil(tmp_box(2).BoundingBox(1))-1)*32 + 1;
            y1 = (ceil(tmp_box(2).BoundingBox(2))-1)*32 + 1;
            x2 = (ceil(tmp_box(2).BoundingBox(1))-1)*32 + (round(tmp_box(2).BoundingBox(3)))*32;
            y2 = (ceil(tmp_box(2).BoundingBox(2))-1)*32 + (round(tmp_box(2).BoundingBox(4)))*32;
            
            patch_image_batch_data(:,:,:,cls) = imresize(test_image(y1:y2,x1:x2,:),[patch_size(1) patch_size(2)]);
        end
        
        %use the local prediction network
        tic; bb_outputs = net.forward({patch_image_batch_data}); pertime=toc;
        usedtime=usedtime+pertime; avgtime=usedtime/j;
        
        local_result{i,j}=bb_outputs{1};
        %%
        
    end
    fprintf('Average time of %d video: %f\n',i,avgtime);
end

save('./output/local_result.mat','local_result');
caffe.reset_all();