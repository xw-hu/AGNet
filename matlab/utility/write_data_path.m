%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data Path
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

clear all; close all;

% choose which data list to generate
 dataType = 'test';
% dataType = 'train';

% set root_dir to your directory
root_dir = './data/';

if strcmp(dataType,'test')
    root_dir = strcat(root_dir,'test_image/');
else
    root_dir = strcat(root_dir,'train_image/');
end

image_dir = [root_dir 'tool_video_'];

file_name = sprintf('./data/tool_detection_%s.txt',dataType);
fid = fopen(file_name, 'wt');

if(strcmp(dataType,'test'))
    
    groud_truth = './data/test_label';
    for i=11:15
        file_name = strcat('/tool_video_',int2str(i));
        
        ground_truth_file = strcat(strcat(groud_truth,file_name),'.txt');
        [gt, toolNames] = ReadToolAnnotationFile(ground_truth_file);
        
        fileFolder = strcat(image_dir,int2str(i));
        full_name = strcat(strcat(fileFolder,file_name),'_');
       
        for j=1:size(gt,1)
            final_jpg = strcat(full_name,int2str(gt(j,1)));
            final_jpg = strcat(final_jpg,'.jpg');
            fprintf(fid, '%s', final_jpg);
            
            for k=2:size(gt,2)
                fprintf(fid, ' %d', gt(j,k));
            end
            fprintf(fid, ' \n');
        end
    end
else if(strcmp(dataType,'train'))
   
    groud_truth = './data/train_label';
    for i=1:10
        
        if(i<10)
            file_name = strcat('/tool_video_0',int2str(i));
        else
            file_name = strcat('/tool_video_',int2str(i));
        end
        
        ground_truth_file = strcat(strcat(groud_truth,file_name),'.txt');
        [gt, toolNames] = ReadToolAnnotationFile(ground_truth_file);
        
        fileFolder = strcat(image_dir,int2str(i));
        full_name = strcat(strcat(fileFolder,file_name),'_');
       
        for j=1:size(gt,1)
            final_jpg = strcat(full_name,int2str(gt(j,1)));
            final_jpg = strcat(final_jpg,'.jpg');
            fprintf(fid, '%s', final_jpg);
            
            for k=2:size(gt,2)
                fprintf(fid, ' %d', gt(j,k));
            end
            fprintf(fid, ' \n');
        end
    end    
    end
end

fclose(fid);




