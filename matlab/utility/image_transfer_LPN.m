function [image_data] = image_transfer(image_data, do_mirror, image_size)

   image_data = imresize(image_data, image_size);
   
%    h_off = floor(rand(1)*(image_size(1)-crop_size(1)+1));  %0-32
%    w_off = floor(rand(1)*(image_size(2)-crop_size(2)+1)); 
%    
%    image_data = image_data(h_off+1:crop_size(1)+h_off,w_off+1:crop_size(2)+w_off,:);
   
   if(do_mirror && rand(1)<0.5) %flip left and right
       for k=1:3
           image_data(:,:,k) = fliplr(image_data(:,:,k));
       end
   end 
   
   if(do_mirror && rand(1)<0.5) %flip up and down
       for k=1:3
           image_data(:,:,k) = flipud(image_data(:,:,k));
       end
   end 
   
   %%%%%%% for each image

   image_data = permute(image_data, [2 1 3]);

end