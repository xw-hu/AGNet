function [ pred ] = ReadToolPredictionFile( pred_file )
%READTOOLPREDICTIONFILE 
% This function reads the tool prediction file (result)

fid_pred = fopen(pred_file, 'r');

% read the header first
fgets(fid_pred); 

% read the labels
pred = textscan(fid_pred, '%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f');
pred = horzcat(pred{:});

end

