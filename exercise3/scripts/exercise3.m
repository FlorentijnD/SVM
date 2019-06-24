clear
%close all
sig = [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000,2000000,5000000,10000000,20000000,50000000];
entropy = zeros(1,size(sig,2))
converge = zeros(1,size(sig,2))
updates = zeros(10,size(sig,2),100);
whenupdates = updates;
for k = 1:10
    i=1;
    for sig2 = sig
    X = 3.*randn(100,2);
    ssize = 10;
    subset = zeros(ssize,2);
        con=0;
        for t = 1:100,

          %
          % new candidate subset
          %
          r = ceil(rand*ssize);
          candidate = [subset([1:r-1 r+1:end],:); X(t,:)];

          %
          % is this candidate better than the previous?
          %
          if kentropy(candidate, 'RBF_kernel',sig2)>...
                kentropy(subset, 'RBF_kernel',sig2),
            subset = candidate;
            con=t;
            updates(k,i,t)=1;
            whenupdates(k,i,t) = t;
          end

          %
          % make a figure
          %
          plot(X(:,1),X(:,2),'b*'); hold on;
          plot(subset(:,1),subset(:,2),'ro','linewidth',6); hold off; 
        end
        converge(i)=con;
        entropy(i)=kentropy(subset, 'RBF_kernel',sig2);
        i=i+1
    end
end
%%
semilogx(sig,entropy)
ylabel("entropy",'FontSize',18)
xlabel("\sigma^2",'FontSize',18)
figure 
semilogx(sig,mean(updates,[1 3])*100)
ylabel("average number of updates",'FontSize',18)
xlabel("\sigma^2",'FontSize',18)
semilogx(sig,mean(whenupdates,[1 3]))
ylabel("average iteration of updates",'FontSize',18)
xlabel("\sigma^2",'FontSize',18)

 %% Homework 1

 load digits;
 clear size ;

 [N, dim]=size(X);

 Ntest=size(Xtest1,1);
 
 noise = 0.3;

 Xn = X;
 for i=1:N
 randn('state', i);
 Xn(i,:) = X(i,:) + noise*randn(1, dim);
 end

 Xnt = Xtest1;
 for i=1:size(Xtest1,1)
 randn('state', N+i);
 Xnt(i,:) = Xtest1(i,:) + noise*randn(1,dim);
 end

 Xtr = X(1:1:end,:);
 
 sig2 = dim*mean(var(Xtr));
 sigmafactor = 10.^linspace(-2.5,1,40);
 sig2_span=sig2.*sigmafactor;

 % which number of eigenvalues of kpca
 npcs = 64;
 lpcs = length(npcs);
 
 score1 = zeros(length(sig2_span),1) ;
 score2 = zeros(length(sig2_span),1) ;

 for idx_sig2 = 1:length(sig2_span)
     sig2 = sig2_span(idx_sig2) ;

     disp('Kernel PCA: extract the principal eigenvectors in feature space');
     disp(['sig2 = ', num2str(sig2)]);

     [lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240);
     [lam, ids]=sort(-lam); lam = -lam; U=U(:,ids);

     ker = kernel_matrix(Xtr,'RBF_kernel',sig2,eye(dim))';

     % KPCA
     disp(' ');
     disp(' Denoise using the first PCs');
     % choose the digits for test
     digs=[0:9]; ndig=length(digs);
     m=2; % Choose the mth data for each digit

     Xdt=zeros(ndig,dim);
     Xtt=zeros(size(X,1),dim);

     disp(['nb_pcs = ', num2str(npcs)]);
     Ud=U(:,(1:npcs)); lamd=lam(1:npcs);

     for i=1:ndig
         dig=digs(i);
         fprintf('digit %d : ', dig)
         xt=Xnt(i,:);
         Xdt(i,:) = preimage_rbf(Xtr,sig2,Ud,xt,'denoise');
     end % for i
     score1(idx_sig2,k) = sqrt(mean((mean(Xdt-Xtest1,2).^2)));
     for i=1:size(X,1)
         xt=Xn(i,:);
         Xtt(i,:) = preimage_rbf(Xtr,sig2,Ud,xt,'denoise');
     end % for i
      score2(idx_sig2,k) = sqrt(mean((mean(Xtt-Xtr,2).^2)));
      Xtt
 end % for k

 [~,idx_min] = min(score1) ;
 sig2_opt = sig2_span(idx_min) ;
 
figure
 loglog(sig2_span,score1,'r',sig2_span,score2,'b');
 ylim([0.001 1]);
 title({"RMSME for noise factor="+noise, " and "+npcs+" principal components"},'FontSize',18);
 xlabel("\sigma^2",'FontSize',18);
 ylabel('RMSME','FontSize',18) ;
 legend("validation set","training set");
 %%
 close all
 clear all
 data = load('shuttle.dat','-ascii');
 data = data(1:10000,:);
 rng('default') ;
 sizeTraining = size(data,1)*0.8;
 indicesTr = randperm(size(data,1),sizeTraining) ;
 X = data(indicesTr,1:end-1);
 Y = data(indicesTr,end);
 Y(Y>1)=-1;
 testX = data(:,1:end-1);
 testY = data(:,end);
 testY(testY>1)=-1;
 testX(indicesTr,:) = [] ;
 testY(indicesTr) = [] ;
 
 
 %Parameter for input space selection
 %Please type >> help fsoperations; to get more information

 k = 10;
 function_type = 'c';
 kernel_type = 'lin_kernel'; % or 'lin_kernel', 'poly_kernel'
 global_opt = 'csa'; % 'csa' or 'ds'

 %Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);
%% cali
% data = load('breast_cancer_wisconsin_data.mat','-ascii'); function_type = 'c';
% data = load('shuttle.dat','-ascii'); function_type = 'c';  data = data(1:2000,:);
data = load('california.dat','-ascii'); function_type = 'f';
% addpath('../LSSVMlab')
data = normalize(data,1) ;

sizeTraining = size(data,1)*0.8;
 indicesTr = randperm(size(data,1),sizeTraining) ;
 X = data(indicesTr,1:end-1);
 Y = data(indicesTr,end);
 testX = data(:,1:end-1);
 testY = data(:,end);
 testX(indicesTr,:) = [] ;
 testY(indicesTr) = [] ;

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 4;
% function_type = 'c'; %'c' - classification, 'f' - regression  
kernel_type = 'poly_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);