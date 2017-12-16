%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ensamble results (Gate Function)
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

load('./output/local_result.mat');
load('./output/global_result.mat');

ground_truth_files = {'./data/test_label/tool_video_11.txt', ...
    './data/test_label/tool_video_12.txt', ...
    './data/test_label/tool_video_13.txt', ...
    './data/test_label/tool_video_14.txt', ...
    './data/test_label/tool_video_15.txt'};

image_dir = ['./data/test_image/' 'tool_video_'];

final_res = zeros(1,7);

for i = 1:length(ground_truth_files)
    ground_truth_file = ground_truth_files{i};
    
    pred_file = [ground_truth_file(1:end-4) '_pred.txt'];
    fid = fopen(pred_file, 'wt');
    
    fprintf(fid, '%s', 'Frame');
    [gt, toolNames] = ReadToolAnnotationFile(ground_truth_file);
    
    for num=1:length(toolNames)
        fprintf(fid, '\t%s', toolNames{num});
    end
    fprintf(fid, '\n');
    
    fileFolder = strcat(image_dir,int2str(i+10));  %test
    file_name = strcat('/tool_video_',int2str(i+10));
         
    full_name = strcat(strcat(fileFolder,file_name),'_');
    
      usedtime = 0; avgtime = 0;
      for j=1:size(gt,1)

            res = global_result{i}(j,:);
            bb_outputs = local_result{i,j};            
            bb_outputs = 1./(1+exp(-bb_outputs));                    
            res = 1./(1+exp(-res));
       
            for n=1:7
                [val, pos] = max(bb_outputs(:,n));
                
                final_res(n) = res(n).*bb_outputs(n,n); 
            end

            fprintf(fid, '%d\t%f %f %f %f %f %f %f\n', gt(j,1),final_res(1),final_res(2),final_res(3),final_res(4),final_res(5),final_res(6),final_res(7));
           
      end
      fprintf('Average time of %d video: %f\n',i,avgtime);
      fclose(fid);
end

calculate_tool_ap(ground_truth_files);