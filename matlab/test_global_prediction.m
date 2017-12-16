%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Global Prediction Network Test
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

weight='./output/global_prediction.caffemodel';
model='./models/global_predcition/deploy.prototxt';

net=caffe.Net(model,weight,'test');

ground_truth_files = {'./data/test_label/tool_video_11.txt', ...
    './data/test_label/tool_video_12.txt', ...
    './data/test_label/tool_video_13.txt', ...
    './data/test_label/tool_video_14.txt', ...
    './data/test_label/tool_video_15.txt'};

image_dir = ['./data/test_image/' 'tool_video_'];

for i = 1:length(ground_truth_files)
    
    ground_truth_file = ground_truth_files{i};
    
    pred_file = [ground_truth_file(1:end-4) '_pred.txt'];
    [gt, toolNames] = ReadToolAnnotationFile(ground_truth_file);
    
    fileFolder = strcat(image_dir,int2str(i+10));  %test
    file_name = strcat('/tool_video_',int2str(i+10));
    
    %%for validation set
    %     if(i<10)
    %         file_name = strcat('/tool_video_0',int2str(i));
    %
    %     else
    %         file_name = strcat('/tool_video_',int2str(i));
    %     end
    
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
        
        tic; outputs = net.forward({test_image}); pertime=toc;
        usedtime=usedtime+pertime; avgtime=usedtime/j;
        
        res = squeeze(outputs{1});
        
        % if visualize = 1;
        % visualize the bounding box predicted by attention map 
        % (the boundbox with the maximum predicted category)
        visualize = 0; 
        if visualize
            cls_feat = net.blobs('cls_conv').get_data();
            
            [max_v, max_p] = max(res);
            
            feature_cls = (cls_feat(:,:,max_p));
            level = multithresh(feature_cls);
            BW = imquantize(feature_cls,level);
            tmp_box = regionprops(BW,'BoundingBox');
            
            postive_bb(1) = (ceil(tmp_box(2).BoundingBox(1))-1)*32+1;
            postive_bb(2) = (ceil(tmp_box(2).BoundingBox(2))-1)*32+1;
            postive_bb(3) = (round(tmp_box(2).BoundingBox(3)))*32;
            postive_bb(4) = (round(tmp_box(2).BoundingBox(4)))*32;
            x1= postive_bb(1);
            y1= postive_bb(2);
            x2= postive_bb(1)+postive_bb(3)-1;
            y2= postive_bb(2)+postive_bb(4)-1;
            
            im = imread(final_jpg);
            im = imresize(im,[imgH imgW]);
            im = permute(im, [2 1 3]);
            
            fg = figure;
            imshow(im,'border','tight');
            hold on;
            rectangle('Position', [x1 y1 x2-x1+1 y2-y1+1], 'EdgeColor', 'r', ...
                'LineWidth', 3);
        end
        
        global_result{i}(j,:)=res(:);

    end
    fprintf('Average time of %d video: %f\n',i,avgtime);
end

save('./output/global_result.mat','global_result');
caffe.reset_all();