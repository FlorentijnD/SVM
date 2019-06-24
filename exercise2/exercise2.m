demofun
%% 1.1
uiregress
rng(1)

X = (-3:0.01:3)';
Y = sinc (X) + 0.1.* randn ( length (X), 1);

Xtrain = X (1:2: end);
Ytrain = Y (1:2: end);
Xtest = X (2:2: end);
Ytest = Y (2:2: end);
%% 1.2
close all
gam = 10000;
sig2 = 100;
type = 'function estimation';
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
Yt = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
rmse = sqrt(mean((Yt-Ytest).^2))
figure
ylim([-0.5 1.2])
plot(Xtest, Yt)
ylim([-0.5 1.2])
hold on
plot(Xtest,Ytest,'.')
hold off
title({"Function approximation and test set", "using LS-SVM^R^B^F_{\sigma^2=}_{"+sig2+"} _{\gamma=}_{"+gam+"}"}, 'FontSize',18)

figure
ylim([-0.5 1.2])
plot(Xtest,Yt)
ylim([-0.5 1.2])
hold on
plot(Xtrain,Ytrain,'.')
hold off
title({"Function approximation and training set", "using LS-SVM^R^B^F_{\sigma^2=}_{"+sig2+"} _{\gamma=}_{"+gam+"}"}, 'FontSize',18)
%%
time=zeros(1,30)
gamm=time;
sigg=time;
costt=time;
for i=1:30
    tic;
    [gam,sig2,cost]=tunelssvm({Xtrain,Ytrain,'f',[],[],'RBF_kernel'},'gridsearch','crossvalidatelssvm',{10,'mse'});
    time(i)=toc;    
    gamm(i)=gam;
    sigg(i)=sig2;
    costt(i)=cost; 
end
mean(time(1:30))
var(time(1:30))
sqrt(mean(costt(1:30)))
sqrt(var(costt(1:30)))
%%
time=zeros(1,30)
gamm=time;
sigg=time;
costt=time;
for i=1:30
    tic;
    [gam,sig2,cost]=tunelssvm({Xtrain,Ytrain,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mse'});
    time(i)=toc;    
    gamm(i)=gam;
    sigg(i)=sig2;
    costt(i)=cost; 
end
mean(time(1:30))
var(time(1:30))
sqrt(mean(costt(1:30)))
sqrt(var(costt(1:30)))
mean(gamm)
var(gamm)
mean(sigg)
var(sigg)
%% bayesian framework
gam = 10 % initial values are important
sig2 = 0.4 % initial values are important
[~, alpha ,b] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 1);
[~, gam] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 2);
[~, sig2 ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 3);

sig2e = bay_errorbar ({ Xtrain , Ytrain , 'f', gam , sig2 }, 'figure')

sig = [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000,2000000,5000000,10000000,20000000,50000000];
ga = [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000,2000,5000];

i=1;

gamma = 0;
if gamma==0
    crit_L1=zeros(size(ga));
    crit_L2=crit_L1;
    crit_L3=crit_L1;
    for gam=ga
        crit_L1(i) = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 1)
        crit_L2(i) = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 2)
        crit_L3(i) = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 3)
        i=i+1;
    end
    
    figure
    semilogx(ga,crit_L2)
    xlabel("\gamma")
    ylabel({"Cost proportional to the posterior", "on the second level"},'FontSize',18)
    title("Cost for different \gamma values, for \sigma^2="+sig2,'FontSize',18)
else
    crit_L1=zeros(size(sig));
    crit_L2=crit_L1;
    crit_L3=crit_L1;
    for sig2=sig
        crit_L1(i) = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 1)
        crit_L2(i) = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 2)
        crit_L3(i) = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 3)
        i=i+1;
    end
    figure
    semilogx(sig,crit_L3)
    xlabel("\sigma^2")
    ylabel({"Cost proportional to the posterior", "on the third level"},'FontSize',18)
    title("Cost for different \sigma^2 values, for \gamma="+gam,'FontSize',18)
end
% "Cost proportional to the posterior on the second level
% By taking the negative logarithm of the posterior and neglecting all constants, one
%   obtains the corresponding cost. Computation is only feasible for
%   one dimensional output regression and binary classification
%   problems. Each level has its different in- and output syntax."
%% 1.3 ARD
gam = 10 % initial values are important
sig2 = 0.4 % initial values are important
[~, alpha ,b] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 1);
[~, gam] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 2);
[~, sig2 ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 3);


X = 6.* rand (100 , 3) - 3;
Y = sinc (X(: ,1)) + 0.1.* randn (100 ,1) ;
[ selected , ranking ] = bay_lssvmARD ({X, Y, 'f', gam , sig2 });

[out,idx] = sort(X(:,1))
plot(out,Y(idx))
hold on
plot(X(:,2),Y,'.')
plot(X(:,3),Y,'.')
legend("first input dimension","second input dimension","third input dimension")
hold off
%%
rng('default');
meanmse = zeros(1,7)
meanmse(1)=crossvalidate({X(:,1),Y,'f',gam,sig2,'RBF_kernel'}) % output is mean mse
meanmse(2)=crossvalidate({X(:,2),Y,'f',gam,sig2,'RBF_kernel'}) % output is mean mse
meanmse(3)=crossvalidate({X(:,3),Y,'f',gam,sig2,'RBF_kernel'}) % output is mean mse
meanmse(4)=crossvalidate({X(:,[1,2]),Y,'f',gam,sig2,'RBF_kernel'}) % output is mean mse
meanmse(5)=crossvalidate({X(:,[1,3]),Y,'f',gam,sig2,'RBF_kernel'}) % output is mean mse
meanmse(6)=crossvalidate({X(:,[2,3]),Y,'f',gam,sig2,'RBF_kernel'}) % output is mean mse
meanmse(7)=crossvalidate({X,Y,'f',gam,sig2,'RBF_kernel'}) % output is mean mse
%% 1.4 Robust regression
X = ( -6:0.2:6)';
Y = sinc (X) + 0.1.* rand ( size (X));
% Outliers can be added via:
out_a = [15 17 19];
Y( out_a) = 0.7+0.3* rand ( size ( out_a));
out_b = [41 44 46];
Y( out_b) = 1.5+0.2* rand ( size ( out_b));
%Let's say we first train a LS-SVM regressor model, without giving special attention to the
%outliers. We can implement this as:
model = initlssvm (X, Y, 'f', [], [], 'RBF_kernel');
costFun = 'crossvalidatelssvm';
model = tunelssvm (model , 'simplex', costFun , {10 , 'mse';});
plotlssvm ( model );
%We explicitly write out the code here, since the object oriented notation is used.
%If we were to train a robust LS-SVM model, using robust crossvalidation, we can imple-
%ment this as:
%%
model = initlssvm (X, Y, 'f', [], [], 'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
wFun = 'whuber';
model = tunelssvm (model , 'simplex', costFun , {10 , 'mae';}, wFun );
model = robustlssvm ( model );
figure
plotlssvm ( model );
%% 1.4 own interpretation
[gam,sig2,cost] = tunelssvm ({X, Y, 'f', [], [], 'RBF_kernel'} , 'simplex', 'crossvalidatelssvm' , {10 , 'mse';});
[alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'}) ;

figure;
plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel'},{alpha,b});
hold on ;
scatter(X,Y,10,'filled','g') ;
scatter(X([out_a,out_b]),Y([out_a,out_b]),10,'filled','b') ;
hold off;
figure;
plot(alpha,'.g')
hold on
plot([out_a,out_b],alpha([out_a,out_b]),'.b');
xlabel("data index", 'FontSize',18)
ylabel("alpha", 'FontSize',18)
%% 'whuber', 'whampel', 'wlogistic' and 'wmyriad'
model = initlssvm (X, Y, 'f', [], [], 'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
wFun = 'wmyriad';
model = tunelssvm (model , 'simplex', costFun , {10 , 'mae';}, wFun );
model = robustlssvm ( model );

figure;
plotlssvm ( model );
hold on ;
scatter(X,Y,10,'filled','g') ;
scatter(X([out_a,out_b]),Y([out_a,out_b]),10,'filled','b') ;
hold off;
figure;
plot(model.alpha,'.g')
hold on
plot([out_a,out_b],model.alpha([out_a,out_b]),'.b');
xlabel("data index", 'FontSize',18)
ylabel("alpha", 'FontSize',18)

%% Homework problems 
%% 1
load logmap.mat

order = 2;
X = windowize(Z, 1:( order + 1));
Y = X(:, end);
X = X(:, 1: order );
 
gam = 10;
sig2 = 10;
[alpha , b] = trainlssvm ({X, Y, 'f', gam , sig2 });
 
Xs = Z(end - order +1: end , 1);
 
nb = 50;
prediction = predict ({X, Y, 'f', gam , sig2 }, Xs , nb);

rmse = sqrt(mean((Ztest-prediction).^2))
figure ;
hold on;
plot (Ztest , 'k');
plot ( prediction , 'r');
title("auto regressive model for \sigma^2="+sig2+", \gamma="+gam+". RMSE="+rmse)
x0=10;
y0=10;
width=800;
height=200;
set(gcf,'position',[x0,y0,width,height])
hold off;
%%
orders = 2:1:50
rmsee = ones(1,size(orders,2))
gamm=rmse
sigg=rmse
i=1
rmsemax=1;
for order = orders
    X = windowize(Z, 1:( order + 1));
    Y = X(:, end);
    X = X(:, 1: order );
    Xs = Z(end - order +1: end , 1);
    nb = 50;

    for j =1:10
        [gam,sig2,cost] = tunelssvm ({X, Y, 'f', [], [], 'RBF_kernel'} , 'simplex', 'crossvalidatelssvm' , {10 , 'mse';});
        [alpha , b] = trainlssvm ({X, Y, 'f', gam , sig2 });
        prediction = predict ({X, Y, 'f', gam , sig2 }, Xs , nb);
        rmse=sqrt(mean((Ztest-prediction).^2))
        if rmse<rmsee(i)
            rmsee(i)=rmse;
            gamm(i)=gam;
            sigg(i)=sig2;
        end
        if rmse<rmsemax
            close all
            figure ;
            hold on;
            plot (Ztest , 'k');
            plot ( prediction , 'r');
            title("auto regressive model for \sigma^2="+sig2+", \gamma="+gam+". RMSE="+rmse)
            x0=10;
            y0=10;
            width=800;
            height=200;
            set(gcf,'position',[x0,y0,width,height])
            hold off;
            rmsemax=rmse;
        end
        
    end
    i=i+1
end
%%
plot(2:50,sigg)
ylabel("RMSE",'FontSize',18)
xlabel("order",'FontSize',18)
%%
index = find(rmsee==min(rmsee))
sig2=sigg(index)
gam=gamm(index)
[alpha , b] = trainlssvm ({X, Y, 'f', gam , sig2 });

Xs = Z(end - order +1: end , 1);
 
nb = 50;
prediction = predict ({X, Y, 'f', gam , sig2 }, Xs , nb);

rmse = sqrt(mean((Ztest-prediction).^2))
figure ;
hold on;
plot (Ztest , 'k');
plot ( prediction , 'r');
title("auto regressive model for \sigma^2="+sig2+", \gamma="+gam+". RMSE="+rmse)
x0=10;
y0=10;
width=800;
height=200;
set(gcf,'position',[x0,y0,width,height])
hold off;
%%
load santafe.mat
orders = 1:1:60
rmsee = inf(1,size(orders,2))
gamm=rmsee
sigg=rmsee
i=1
rmsemax=inf;
for order = orders
    X = windowize(Z, 1:( order + 1));
    Y = X(:, end);
    X = X(:, 1: order );
    Xs = Z(end - order +1: end , 1);
    nb = 200;

    for j =1:10
        [gam,sig2,cost] = tunelssvm ({X, Y, 'f', [], [], 'RBF_kernel'} , 'simplex', 'crossvalidatelssvm' , {10 , 'mse';});
        [alpha , b] = trainlssvm ({X, Y, 'f', gam , sig2 });
        prediction = predict ({X, Y, 'f', gam , sig2 }, Xs , nb);
        rmse=sqrt(mean((Ztest-prediction).^2))
        if rmse<rmsee(i)
            rmsee(i)=rmse;
            gamm(i)=gam;
            sigg(i)=sig2;
        end
        if rmse<rmsemax
            close all
            figure ;
            hold on;
            plot (Ztest , 'k');
            plot ( prediction , 'r');
            title("auto regressive model for \sigma^2="+sig2+", \gamma="+gam+". RMSE="+rmse)
            x0=10;
            y0=10;
            width=800;
            height=200;
            set(gcf,'position',[x0,y0,width,height])
            hold off;
            rmsemax=rmse;
        end
        
    end
    i=i+1
end
%%
%%
plot(1:60,rmsee)
ylabel("RMSE",'FontSize',18)
xlabel("order",'FontSize',18)