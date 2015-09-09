clear all;
close all;

addpath('/home/ghooo/libsvm-3.18/matlab/');

Y = -1:0.1:1;
X = -1:0.1:1;
Y = Y';
X = X';

% svm options
svmopts=['-s 4 -t 0 -c 2 -g 1'];

% train SVM
model=svmtrain(Y, X, svmopts);

% test SVM on training data
[Yout, Acc, Yext]=svmpredict(Y,X,model,'');

tY = ones(1,1);
tX = [0.123];
% test SVM on test data
[tYout, tAcc, tYext]=svmpredict(tY,tX,model,'');
tYout
