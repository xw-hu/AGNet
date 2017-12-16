%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test AGNet
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

fprintf('Test Global Prediction Network\n');
test_global_prediction;

fprintf('Test Local Prediction Network\n');
test_local_prediction;

result_ensamble;
