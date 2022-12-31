clc
clear 
close all
%{
%Importing images addresses
imds = imageDatastore("Dataset",...
    "IncludeSubfolders",true,"LabelSource","foldernames","FileExtensions",[".jpg",".png"]);
%Dividing Test  and Train Images
[n,m]=size(imds.Files);
%Seprating Mask and Original Images
for t=1:n
    my_field = strcat('v',num2str(t));
    z=imds.Files(t,1);
    z=char(z);
    variable.(my_field) = imread(z);
    IMG=variable.(my_field);
    AllReals.(my_field)=IMG(:,1:256,:);
    AllMasks.(my_field)=IMG(:,257:end,:);
end
%Only Tests
for t1=1:200
    mytest=strcat('v',num2str(t1));
    Test_Reals.(mytest)=AllReals.(mytest);
    Test_Masks.(mytest)=AllMasks.(mytest);
end
%Only Trains
for t2=201:n
    mytrains=strcat('v',num2str(t2));
    Train_Reals.(mytrains)=AllReals.(mytrains);
    Trains_Masks.(mytrains)=AllMasks.(mytrains);
end
%}

%% Storing
%{
%Train Reals
ImageFolder ='Train images';
for i=1:1500
    J=(i+200);
    name=strcat('v',num2str(J));
   realTr=Train_Reals.(name); 
     imgName = [ImageFolder,'\Train0',num2str(i),'.jpg'] ;
    imwrite(realTr,imgName) ;
end
%Test Reals
ImageFolder ='Test images';
for i=1:200
    name=strcat('v',num2str(i));
    realTe=Test_Reals.(name); 
     imgName = [ImageFolder,'\Test0',num2str(i),'.jpg'] ;
    imwrite(realTe,imgName) ;
end

%Train Masks
ImageFolder ='Train masks';
for i=1:1500
    J=(i+200);
    name=strcat('v',num2str(J));
    Maskedl=Trains_Masks.(name);
     imgName = [ImageFolder,'\Masks',num2str(i),'.jpg'] ;
    imwrite(Maskedl,imgName) ;
end

%Test Masks
ImageFolder ='Test masks';
for i=1:200
    name=strcat('v',num2str(i));
    unmasked=Test_Masks.(name);
     imgName = [ImageFolder,'\TestM',num2str(i),'.jpg'] ;
    imwrite(unmasked,imgName) ;
end
%}
%%
%All Masks
%{
ImageFolder ='All Masks';
for i=1:1500
   J=(i+200);
    name=strcat('v',num2str(J));
    Maskedl=Trains_Masks.(name);
    imgName = [ImageFolder,'\Masks',num2str(J),'.jpg'] ;
    imwrite(Maskedl,imgName) ;
end
for i=1:200
    name=strcat('v',num2str(i));
    unmasked=Test_Masks.(name);
    imgName = [ImageFolder,'\Masks',num2str(i),'.jpg'] ;
    imwrite(unmasked,imgName) ;
end

%}

%% Labeling Masks with diferrent classes
%{

%Masks Datastore
dataDir='All Masks';
Masks_DIR= fullfile(dataDir);
Masks_imd=imageDatastore(Masks_DIR);


%Seprable
ImageFolder ='Test SEP L';
for i=1:1:200
    Maskedl=imread(Masks_imd.Files{i,1});
    L = imsegkmeans(Maskedl,11);
    kkk=uint8(L);
     %An Example
      % B = labeloverlay(Maskedl,kkk); 
       %figure;imshow(B);

    imgName = [ImageFolder,'\',num2str(i),'.png'] ;
    imwrite(kkk,imgName) ;
end

ImageFolder ='Train SEP L';
for i=201:1:1700
    J=i-200;
    Maskedl=imread(Masks_imd.Files{i,1});
    L = imsegkmeans(Maskedl,11);
    kkk=uint8(L);
     %An Example
      % B = labeloverlay(Maskedl,kkk); 
       %figure;imshow(B);

    imgName = [ImageFolder,'\',num2str(J),'.png'] ;
    imwrite(kkk,imgName) ;
end




%All together
Masks=imtile(Masks_imd,'Gridsize',[34,50]);
%YCCmask=rgb2ycbcr(Masks);
%{
%ur=Masks(1:1024,1:1024,:);
%uh=rgb2hsv(ur);
%ul=rgb2lab(ur);
%YCBCR = rgb2ycbcr(ur);
%figure;imshow(ur);figure;imshow(uh);figure;imshow(ul);figure;imshow(YCBCR)
%v=uint8(rgb2hsv(Masks(1:256,1:256,:)));
%NumClasses=12;
%[vl,Centers]=imsegkmeans(v,NumClasses,'NumAttempts',1);
%}
NumClasses=11;
[mylab,Centers]=imsegkmeans(Masks,NumClasses);

a=1;
b=1;
c=256;
d=256;
v=1;
g=1;
num=1;
for w=1:34
    for z=1:50
e=1;
f=1;
for i=a:1:c
    for j=b:1:d
     pixel= mylab(i,j);  
     Blocks(e,f)=pixel;
     f=f+1;
    end
    f=1;
    e=e+1;
end


ImageFolder ='';
name=strcat('v',num2str(num));
kkk=uint8(Blocks);
     imgName = [ImageFolder,'\',num2str(num),'.png'] ;
    imwrite(kkk,imgName) ;
    num=num+1;

%figure;imshow(uint8(Blocks));
g=v+z;
d=d+256;
b=b+256;
    end
    b=1;
    d=256;
    c=c+256;
    a=a+256;
    v=v+50;
end

%Real Train Inputs
dataDir='Train images';
Train_DIR= fullfile(dataDir);
Train_imd=imageDatastore(Train_DIR);
%Masks Trains 
dataDir='Train masks';
Train_Mask_DIR= fullfile(dataDir);
Train_Mask_imd=imageDatastore(Train_Mask_DIR);

%Real Test inputs
dataDir='Test images';
Test_DIR= fullfile(dataDir);
Test_imd=imageDatastore(Test_DIR);
%Masks Test 
dataDir='Test masks';
Test_Mask_DIR= fullfile(dataDir);
Test_Mask_imd=imageDatastore(Test_Mask_DIR);

% Trains Labels
dataDir='Train Labels';
Train_Labels_DIR= fullfile(dataDir);
Train_Labels_imd=imageDatastore(Train_Labels_DIR);
% Test Labels
dataDir='Test Labels';
Test_DIR= fullfile(dataDir);
Test_Labels_imd=imageDatastore(Test_DIR);

%An Example
oo=imread(Train_imd.Files{5, 1});
ss=imread(Train_Labels_imd.Files{5, 1});
B = labeloverlay(oo,ss); 
figure;imshow(B);



oo=imread(Train_imd.Files{10, 1});
ss=imread(Train_Labels_imd.Files{10, 1});
B = labeloverlay(oo,ss); 
figure;imshow(B);

%%
classNames = ["dynamic" ,"road" , "tree", "walkroad","null", "legs", "car" ,...
                         "building", "sign" ,  "sky", "other"];
ids=1:11;
pxds = pixelLabelDatastore(Train_Labels_DIR,classNames,ids);
ds = combine(Train_imd,pxds);
%%
[row,col,dep]=size(oo);
imageSize = ([row,col,dep]);
numClasses = 11;
encoderDepth=3;
lgraph = unetLayers(imageSize,numClasses,'EncoderDepth',encoderDepth);
options = trainingOptions('sgdm', ...
    'Minibatchsize',1 ,...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',10, ...
    'shuffle','never',...
    'VerboseFrequency',2,...
    'Plots','training-progress');
%'Plots','training-progress'
net = trainNetwork(ds,lgraph,options);

pwd='Outputs';
pxdsResults = semanticseg(Test_imd,net, ...
    'MiniBatchSize',1, ...
    'WriteLocation',pwd, ...
    'Verbose',false);
%pxdsTest.ReadFcn=@(x) imresize(imread(x), [32 32]);


pxdstest = pixelLabelDatastore(Test_DIR,classNames,ids);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdstest,'Verbose',false);
out=(metrics.DataSetMetrics);
ACC=out{1,1}*100;
fprintf('\n\n Our Accuracy by using this Network is :%f\n\n',ACC);


%example
org=imread(Test_imd.Files{10, 1});
msk=imread(Test_Mask_imd.Files{10, 1});
a1=imread(pxdsResults.Files{10, 1});
a2=imread(pxdstest.Files{10, 1});

b=imread(Test_imd.Files{10,1});

B2=labeloverlay(b,a2); 
B1 = labeloverlay(b,a1); 



figure;subplot(2,2,1);imshow(org);title('Org')
subplot(2,2,2);imshow(msk);title('Mask')
subplot(2,2,3);imshow(B1);title('Output')
subplot(2,2,4);imshow(B2);title('Input')
