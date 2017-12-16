%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image Alignment
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

function [new_image] = im_alignment(image_data)

%figure(1),imshow(image_data);


% find the no-zero position
BW = imquantize(image_data,5);
mask = uint8((BW(:,:,1)-1).*(BW(:,:,2)-1).*(BW(:,:,3)-1)).*255;


% find the maximum connected region
L = bwlabel(mask);
stats = regionprops(L);
Ar = cat(1, stats.Area);
ind = find(Ar ==max(Ar));

if (isempty(ind) || size(ind,1)~=1)
    new_image = image_data;
    return;
end

mask(find(L~=ind))=0;
mask_g = im2bw(mask);

tmp_box = regionprops(mask_g,'BoundingBox');

new_image = image_data(round(tmp_box.BoundingBox(2)):tmp_box.BoundingBox(4)+round(tmp_box.BoundingBox(2))-1,...
    round(tmp_box.BoundingBox(1)):tmp_box.BoundingBox(3)+round(tmp_box.BoundingBox(1))-1,:);

%figure(2),imshow(new_image);

end