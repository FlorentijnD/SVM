X1 = randn(50,2) + 1;
X2 = randn(50,2) - 1;
Y1 = ones(50,1);
Y2 = ones(51,1);

%% 1.3 : LSSVM
%democlass
%load iris.mat
%load ripley.mat
% load breast.mat
% Xtrain=trainset;
% Ytrain=labels_train;
% Xtest=testset;
% Ytest=labels_test;
% size(Xtrain)
% size(Xtrain(Ytrain==1,:))
load diabetes.mat
Xtrain=trainset;
Ytrain=labels_train;
Xtest=testset;
Ytest=labels_test;
size(Xtrain)
size(Xtrain(Ytrain==1,:))
figure;
hold on
plot(Xtrain(Ytrain==1,1),Xtrain(Ytrain==1,2),'ro')
plot(Xtrain(Ytrain==-1,1),Xtrain(Ytrain==-1,2),'bo')  
hold off
figure;
hold on
plot(Xtest(Ytest==1,1),Xtest(Ytest==1,2),'ro')
plot(Xtest(Ytest==-1,1),Xtest(Ytest==-1,2),'bo')  
hold off
%% polynomial kernel
type='c'; 
gam = 1; 
t = 1; 
degree = 1;
disp('Polynomial kernel of degree 2'),

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'});

figure; plotlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xtest);

err = sum(Yht~=Ytest)/size(Ytest,1)
fp = sum((Yht-Ytest)>0)
fn = sum((Yht-Ytest)<0)
%% RBF / sigma2
gam = 1; sig2list=[0.01,0.03,0.1,0.3, 1, 3, 10, 25, 25];

for sig2=sig2list,
    disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)
    disp('Press any key to continue...'),    pause,    
end
fp = sum((Yht-Ytest)>0)
fn = sum((Yht-Ytest)<0)
%% RBF / gamma
gamlist = [0.01,0.03,0.1,0.3, 1, 3, 10, 30, 100,1000,10000]; sig2=0.3
errlist=[];The      

for gam=gamlist,
    disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)
    disp('Press any key to continue...'),     pause,    
end
fp = sum((Yht-Ytest)>0)
fn = sum((Yht-Ytest)<0)
%% use of validation set
gamlist=[0.001,0.01,0.1,1, 10, 100, 1000];
siglist=gamlist;
perf1=zeros(size(gamlist,2),size(siglist,2));
perf2=perf1;
perf3=perf1;
i=1;
j=1;
for gam = gamlist
    j=1;
    for sig2 = siglist
        perf1(i,j)=rsplitvalidate({Xtrain, Ytrain, 'c', gam, sig2, 'RBF_kernel'},0.75, 'misclass');
        perf2(i,j)=crossvalidate({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},10,'misclass');
        perf3(i,j)=leaveoneout({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},'misclass');
        j=j+1
    end
    i=i+1
end
matvisual(perf1,gamlist,siglist)
matvisual(perf2,gamlist,siglist)
matvisual(perf3,gamlist,siglist)
%% automatic parameter tuning
[gam,sig2,cost]=tunelssvm({Xtrain,Ytrain,'c',[],[],'RBF_kernel'},'gridsearch','crossvalidatelssvm',{10,'misclass'});
[gam,sig2,cost]=tunelssvm({Xtrain,Ytrain,'c',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'misclass'});
[gam,sig2,cost]=tunelssvm({Xtrain,Ytrain,'c',[],[],'poly_kernel'},'simplex','crossvalidatelssvm',{10,'misclass'});
[gam,sig2,cost]=tunelssvm({Xtrain,Ytrain,'c',[],[],'lin_kernel'},'simplex','crossvalidatelssvm',{10,'misclass'});

cost
%% test rbf kernel
[gam,sig2,cost]=tunelssvm({Xtrain,Ytrain,'c',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'misclass'});
%gam = 0.38918;
%sig2= 0.28133;
% gam=43.5308;
% sig2=31.4018;
gam=1614.3283;
sig2=925.67461;
errlist=[];
    disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)
    fp = sum((Yht-Ytest)>0)
fn = sum((Yht-Ytest)<0)
%% test poly kernel
[gam,tdegree,cost]=tunelssvm({Xtrain,Ytrain,'c',[],[],'poly_kernel','preprocess'},'gridsearch','crossvalidatelssvm',{10,'misclass'});
t=tdegree(1);
degree=tdegree(2);
%gam = 6.3019; 
%t = 1.1712; 
%degree = 7;
% gam = 3.595757e-06;
% t = 126.6385;
% degree = 3;
gam = 1.57011e-05; 
t = 24.5254;
degree = 3;
disp('Polynomial kernel of degree 5'),

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'});

figure; plotlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xtest);

err = sum(Yht~=Ytest)/size(Ytest,1)
fp = sum((Yht-Ytest)>0)
fn = sum((Yht-Ytest)<0)
%% test lin kernel
[gam,sig2,cost]=tunelssvm({Xtrain,Ytrain,'c',[],[],'lin_kernel'},'simplex','crossvalidatelssvm',{10,'misclass'});
%gam = 0.039026; 
% gam=0.12886;
gam=1.2869;
disp('linear kernel'),

[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,[],'lin_kernel'});

figure; plotlssvm({Xtrain,Ytrain,'c',gam,[],'lin_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({Xtrain,Ytrain,'c',gam,[],'lin_kernel'}, {alpha,b}, Xtest);

err = sum(Yht~=Ytest)/size(Ytest,1)
fp = sum((Yht-Ytest)>0)
fn = sum((Yht-Ytest)<0)
%%
tic
counter=zeros(100,1);
for i=1:100
    [gam,sig,cost]=tunelssvm({Xtrain,Ytrain,'c',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'misclass'});
    
    if(cost>0.04)
        counter(i)=cost;
    end
end
toc
mean(counter)
var(counter)
%% ROC curves
algorithm = 3
if(algorithm==1)
    gam=0.3892;
    sig2=0.2813;
    gam=43.5308;
    sig2=31.4018;
    gam=1614.3283;
    sig2=925.67461;
    %train
    [alpha,b]=trainlssvm({Xtrain, Ytrain, 'c', gam, sig2, 'RBF_kernel'});
    %classificatin
    [Yest,Ylatent]=simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'}, {alpha, b}, Xtest);
    [Yestt,Ylatentt]=simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'}, {alpha, b}, Xtrain);
    %roc
end
if(algorithm==2)
    gam = 6.3019; 
    t = 1.1712; 
    degree = 7;
    gam = 1; 
    t = 1; 
    degree = 2;
    gam = 3.595757e-06;
    t = 126.6385;
    degree = 3;
    gam = 1.57011e-05; 
    t = 24.5254;
    degree = 3;
    %train
    [alpha,b]=trainlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'});
    %classificatin
    [Yest,Ylatent]=simlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'}, {alpha, b}, Xtest);
    [Yestt,Ylatentt]=simlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'}, {alpha, b}, Xtrain);
    %roc
end
if(algorithm==3)
    gam= 0.039026;
    gam=0.12886;
    gam=1.2869;
    %train
    [alpha, b] = trainlssvm({Xtrain,Ytrain,type,gam,[]   ,'lin_kernel'})
    %classificatin
    [Yest,Ylatent]=simlssvm({Xtrain,Ytrain,type,gam,[]   ,'lin_kernel'}, {alpha, b}, Xtest);
    [Yestt,Ylatentt]=simlssvm({Xtrain,Ytrain,type,gam,[]   ,'lin_kernel'}, {alpha, b}, Xtrain);
    %roc
end
roc(Ylatent,Ytest);
fp = sum((Ylatent-Ytest)>0)
fn = sum((Ylatent-Ytest)<0)
%roc(Ylatentt,Ytrain);
%% Bayesian framework
a=bay_modoutClass({Xtrain,Ytrain,'c',gam,sig2},'figure');
%%
function matvisual(A,gamlist,siglist)
% check the input
if ~isreal(A) || isempty(A) || ischar(A) || ndims(A) > 3
    errordlg('The data array is not suitable for visualization!', ...
             'Error!', 'modal')
    return
end
% determine the matrix size
[M, N, P] = size(A);
% loop through the matrix pages
for p = 1:P
    
    % visualize the matrix page by page
    figure
    himg = imagesc(A(:, :, p));
    colormap jet
    
    % annotation
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 18)
    xlabel('bandwidth')
    ylabel('regularization parameter')
    if P > 1, title(['Matrix page ' num2str(p)]), end
    
    %set(gca, 'XTick', 1:6)
    %set(gca, 'YTick', 1:gamlist(N))
    xticklabels(gamlist)
    yticklabels(siglist)
    hclb = colorbar;
    hclb.Label.String = 'Error rate (%)';
    hclb.Label.FontSize = 18;
    
    for m = 1:M
        for n = 1:N
            text(n, m, num2str(A(m, n, p), 3), ...
                'FontSize', round(6 + 50./sqrt(M.*N)), ...
                'HorizontalAlignment', 'center', ...
                'Rotation', 45)
        end
    end
    
    % set the datatip UpdateFcn
    cursorMode = datacursormode(gcf);
    set(cursorMode, 'UpdateFcn', {@datatiptxt, himg})
end
end
function text_to_display = datatiptxt(~, hDatatip, himg)
% determine the current datatip position
pos = get(hDatatip, 'Position');
% form the datatip label
text_to_display = {['Row: ' num2str(pos(2))], ...
                   ['Column: ' num2str(pos(1))], ...
                   ['Value: ' num2str(himg.CData(pos(2), pos(1)))]};
end