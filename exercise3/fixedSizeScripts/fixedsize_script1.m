clear
close all


X = 3.*randn(100,2);
ssize = 5;
sig2 = 1;
subset = zeros(ssize,2);


indices_subset = 1:ssize;
figure
subplot(1,2,1);
plot(X(:,1),             X(:,2),'b*'); hold on;
plot(X(indices_subset,1),X(indices_subset,2),'ro','linewidth',6); hold off; 
title('original space')

%
% transform the data in feature space
%
features = AFEm(X(indices_subset,:),'RBF_kernel',sig2,X);
subplot(1,2,2);
plot3(features(:,1),             features(:,2),             features(:,3),'k*'); hold on;
plot3(features(indices_subset,1),features(indices_subset,2),features(indices_subset,3),'ro','linewidth',6); hold off;
title('feature space')
features(:,1:3)
features(20:30,1:3)
%%
figure
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
  end
  
  %
  % make a figure
  %
  plot(X(:,1),X(:,2),'b*'); hold on;
  plot(subset(:,1),subset(:,2),'ro','linewidth',6); hold off; 
end
subset
a=zeros(1,5);
for i=1:size(subset,1)
    b=find(X==subset(i,:));
    a(i)=b(1);
end
%%
%
% make a figure
%
figure
subplot(1,2,1);
plot(X(:,1),             X(:,2),'b*'); hold on;
plot(subset(:,1),subset(:,2),'ro','linewidth',6); hold off; 
title('original space')

%
% transform the data in feature space
%
features = AFEm(X(1:10,:),'RBF_kernel',sig2,X);
subplot(1,2,2);
plot3(features(:,1),             features(:,2),             features(:,3),'k*'); hold on;
plot3(features(a,1),features(a,2),features(a,3),'ro','linewidth',6); hold off;
title('feature space')
features(a,1:3)