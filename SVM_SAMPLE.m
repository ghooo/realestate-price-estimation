%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% The American University in Cairo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Sprig 2014 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Pattern Analysis Project %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Mohamed Ghoneim %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 900072605       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CODE DESCRIPTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Sample File That runs SVM

% svm_example - this is a simple example of using LIBSVM utils 
% svmtrain and svmpredict in MATLAB and from the command line
%
% v1.0 Antoni Chan, SVCL, UCSD, Copyright 2006
% v2.0 Jose Costa Pereira, SVCL, UCSD, Copyright 2013

clear all;
close all;

%addpath('libsvm-3.16/matlab/');
%addpath('libsvm-mat-2.82-2/');
addpath('/home/ghooo/libsvm-3.18/matlab/');

randn('seed', 12345);


% specify three classes
mu1 = [-5; 3]; mu2 = [3; 5]; mu3 = [0; -2];
cv1 = [4; 2];  cv2 = [3; 3]; cv3 = [1; 1];

N = 20;
d = 2;
Nt = 10;

% make training set
Y1 = 1*ones(1,N);
X1 = diag(sqrt(cv1))*randn(d, N) + repmat(mu1, 1, N);
Y2 = 2*ones(1,N);
X2 = diag(sqrt(cv2))*randn(d, N) + repmat(mu2, 1, N);
Y3 = 3*ones(1,N);
X3 = diag(sqrt(cv3))*randn(d, N) + repmat(mu3, 1, N);

% make test set
tY1 = 1*ones(1,Nt);
tX1 = diag(sqrt(cv1))*randn(d, Nt) + repmat(mu1, 1, Nt);
tY2 = 2*ones(1,Nt);
tX2 = diag(sqrt(cv2))*randn(d, Nt) + repmat(mu2, 1, Nt);
tY3 = 3*ones(1,Nt);
tX3 = diag(sqrt(cv3))*randn(d, Nt) + repmat(mu3, 1, Nt);

% setup training data
Y = [Y1,Y2,Y3]';
X = [X1,X2,X3];  % each column is a data point
Xt = X';         % each row is a data point

% setup testing data
tY = [tY1, tY2, tY3]';
tX = [tX1, tX2, tX3];
tXt = tX';


% svm options
svmopts=['-c 2 -g 1'];

% train SVM
model=svmtrain(Y, Xt, svmopts);

% test SVM on training data
[Yout, Acc, Yext]=svmpredict(Y,Xt,model,'');

% test SVM on test data
[tYout, tAcc, tYext]=svmpredict(tY,tXt,model,'');

% plot data
figure
hold on
% training data
plot(X1(1,:), X1(2,:), 'rx');
plot(X2(1,:), X2(2,:), 'bx');
plot(X3(1,:), X3(2,:), 'gx');
% test data
cols = 'rbg';
for i=1:3
  ind = find(tYout == i);
  plot(tX(1,ind), tX(2,ind),[cols(i) '.'])
end 
% circle test errors
ind = find(tYout(:) ~= tY(:));
plot(tX(1,ind), tX(2,ind), 'ko');
hold off
legend('C1 training', 'C2 training', 'C3 training', ...
    'test classified C1', 'test classified C2', 'test classified C3', ...
    'test errors','Location','NorthWest');
grid on;

numerrtrain = sum(Yout(:) ~= Y(:))
numerrtest  = sum(tYout(:) ~= tY(:))


