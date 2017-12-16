function [ gt, toolNames ] = ReadToolAnnotationFile( ground_truth_file )
%READTOOLANNOTATIONFILE 
% This function reads the tool annotation file (ground truth)

fid_gt = fopen(ground_truth_file, 'r');

% read the header first
tline = fgets(fid_gt); 
tline = tline(1:end-1);
toolNames = strsplit(tline, '\t');
toolNames(1) = [];

% read the labels
gt = textscan(fid_gt, '%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d');
gt = horzcat(gt{:});

end

