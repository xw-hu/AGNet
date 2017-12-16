function calculate_tool_ap(ground_truth_files)

% load all the data
allGT = [];
allPred = [];
for i = 1:length(ground_truth_files)
    ground_truth_file = ground_truth_files{i};
    pred_file = [ground_truth_file(1:end-4) '_pred.txt'];
    
    [gt, toolNames] = ReadToolAnnotationFile(ground_truth_file);
    pred = ReadToolPredictionFile(pred_file);
    
    if(size(gt, 1) ~= size(pred,1) || size(gt, 2) ~= size(pred,2))
        error(['ERROR:' ground_truth_file '\nGround truth and prediction have different sizes']);
    end
    
    if(~isempty(find(gt(:,1) ~= pred(:,1))))
        error(['ERROR: ' ground_truth_file '\nThe frame index in ground truth and prediction is not equal']);
    end
    
    allGT = [allGT; gt(:,2:end)];
    allPred = [allPred; pred(:,2:end)];
    
    clear gt pred pred_file ground_truth_file;
end

% compute average precision per tool
ap = [];
allPrec = [];
allRec = [];

disp('========================================')
disp('Average precision');
disp('========================================')
for iTool = 1:size(allGT,2)
    matScores = allPred(:,iTool);
    matGT = allGT(:,iTool);    

    % sanity check, making sure it is confidence values
    X = unique(matScores(:));
    if(length(X) == 2)
        disp('- WARNING: the computation of mAP requires confidence values');
    end
    
    % NEW Script - less sensitive to confidence ranges   
    maxScore = max(matScores);
    minScore = min(matScores);
    step = (double(maxScore)-double(minScore))/2000;

    if(minScore == maxScore)
        error('no difference confidence values');
    end

    prec = []; rec = [];
    for iScore = minScore:step:maxScore
        bufScore = matScores > iScore;
        tp = sum(double((bufScore == matGT) & (bufScore == 1)));
        fp = sum(double((bufScore ~= matGT) & (bufScore == 1)));

        if(tp+fp ~= 0)
            rec(end+1) = tp/sum(matGT>0);
            prec(end+1) = tp/(tp+fp);
        end
    end

%     % OLD Script
%     % compute precision/recall
%     [~,si]=sort(-matScores);
%     sortedMatScores = matScores(si);
%     %         sortedMatScores = -sortedMatScores;
%     sortedMatGT = matGT(si);
% 
%     tp=sortedMatGT>0;
%     fp=sortedMatGT<=0;
% 
%     fp=cumsum(fp);
%     tp=cumsum(tp);
%     rec = tp/sum(sortedMatGT>0);
%     prec = tp./(fp+tp);

    % compute average precision - finer way to compute AP
    ap(iTool)=0;
    for t=0:0.1:1
        
        idx = [];
        threshold = 0.00001;
        while (isempty(idx))
            idx = find(abs(rec - t) < threshold);
            threshold = threshold * 2;
        end
        p = mean(prec(idx));

        if isempty(p)
            p=0;
        end
        ap(iTool)=ap(iTool)+p/11;
    end

    disp([toolNames{iTool} ': ' num2str(ap(iTool))]);

end

disp('----------------------------------------')
disp(['All tools: ' num2str(mean(ap))])
disp('----------------------------------------')

end