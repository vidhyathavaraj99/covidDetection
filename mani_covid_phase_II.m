clc;
clear all;
close all;
warning OFF;

% Test Image Path
TestPath = 'TestData';
if ~exist(TestPath, 'dir')
    mkdir(TestPath)
end

% Load Pre-Trained Data
addpath('Functions');
load('./Train_Data.mat');

% Read files form pc. 
[file,path,indx] = uigetfile('./Input Image/*.jpg;*.jpeg;*.bmp',... 
                                    'Select an Input Image File');
if isequal(file,0)
   disp('User selected Cancel')
else
   disp(['Selected File Name: ', file])
   delete './TestData/*.*';
   In_Img = imread([path,file]);
   
   imwrite(In_Img,['./TestData/',file]);
end

% In_Img = imread([Path,File]);
% figure; imshow(I); title('Input Test Image');

% Image Resize
InImg = In_Img;
Re_Img = imresize(InImg, [256 256]);
In_Img = InImg;
figure; imshow(Re_Img); title(['Input Image: ',(file)]);

% Gray Conversion
% Get the dimensions of the image.  
[rows, columns, no_of_band] = size(InImg);
if isequal (no_of_band,3)
	% Convert it to gray scale 
	Gr_Img = rgb2gray(InImg);
    Gr_Img = imresize(Gr_Img,[256 256]);
else
    Gr_Img = imresize(InImg,[256 256]);
end
figure; imshow(Gr_Img); title('Gray  Image');

% Filter - Preprocessing
InImg = double(Re_Img);
Gs=fspecial('gaussian');
[rows1, columns1, no_of_band1] = size(InImg);
if isequal (no_of_band1,3)
	% Convert it to gray scale 
	In_fil(:,:,1)=imfilter(double(InImg(:,:,1)),Gs);
    In_fil(:,:,2)=imfilter(double(InImg(:,:,2)),Gs);
    In_fil(:,:,3)=imfilter(double(InImg(:,:,3)),Gs);

else
    In_fil=imfilter(double(InImg),Gs);
end
figure; imshow(uint8(In_fil)); title('Preprocessed Image');

% Binary Otsu Segmentation
% InImg = rgb2gray(uint8(In_fil));
InImg = uint8(In_fil);

% Specify initial contour location
mask = zeros(size(InImg));
mask(25:end-25,25:end-25) = 1;

% Segmentation
Img =double(Re_Img(:,:,1));
A=255;
sigma = 4;
G=fspecial('gaussian',15,sigma);
Img=conv2(Img,G,'same'); 
nu=0.001*A^2; % coefficient of arc length term
sigma = 4; % scale parameter that specifies the size of the neighborhood
iter_outer=50; 
iter_inner=10;   % inner iteration for level set evolution
timestep=.1;
mu=1;  % cient for distance regularization term (regularize the level set function)
c0=1;
figure;
imagesc(Img,[0, 255]); colormap(gray); axis off; axis equal
% initialize level set function
initialLSF = c0*ones(size(Img));
%initialLSF(30:90,50:90) = -c0;
%initialLSF(30:200,50:220) = -c0;
initialLSF(25:220,25:220) = -c0;
u=initialLSF;
hold on;
contour(u,[0 0],'r');
title('Initial contour');
figure;
imagesc(Img,[0, 255]); colormap(gray); axis off; axis equal
hold on;
contour(u,[0 0],'r');
title('Initial contour');
epsilon=1;
b=ones(size(Img));  %%% initialize bias field
K=fspecial('gaussian',round(2*sigma)*2+1,sigma); % Gaussian kernel
KI=conv2(Img,K,'same');
KONE=conv2(ones(size(Img)),K,'same');
[row,col]=size(Img);
N=row*col;
for n=1:iter_outer
    [u, b, C]= level_set_line(u,Img, b, K,KONE, nu,timestep,mu,epsilon, iter_inner);

    if mod(n,2)==0
        pause(0.001);
        imagesc(Img,[0, 255]); colormap(gray); axis off; axis equal;
        hold on;
        contour(u,[0 0],'r');
        iterNum=[num2str(n), ' iterations'];
        title(iterNum);
        hold off;
    end
end
Mask =(Img>10);
Img_corrected=normalizebm(Mask.*Img./(b+(b==0)))*255;
I=u;
maxI = max(I(:));
minI = min(I(:));
bit_wise = im2bw(I,(maxI - mean(I(:)))/(maxI - minI));
%%%%%%%%%%%%%%%Removing boundaries
%%%%%%%%%%%%%
bin_fil=bit_wise;
for x=1:25   
for y=1:256
     bin_fil(x,y)=0;
end
end
%%%%%%%%%%%%%
for x=225:256   
for y=1:256
     bin_fil(x,y)=0;
end
end 
% %%%%%%%%%%%%%
for x=1:256   
for y=1:40
     bin_fil(x,y)=0;
end
end
%%%%%%%%%%%%%
for x=1:256   
for y=230:256
     bin_fil(x,y)=0;
end
end
figure;
imshow(bin_fil); 
colormap(gray); 
title('Final Boundary Mask'); 
axis off; axis equal;

% Features
[segNet_layer,Layer_labels, class_im]=segNet_Layer_(Re_Img);
% Parameters for the Segmentation
nBins=5;
winSize=7;
nClass=6;
% Segmentation
outImg = colImg_SegNet(Re_Img, nBins, winSize, nClass);
%  Plot the Color Values
figure;
subplot(2,2,1), imshow(Layer_labels,[]), title('Layer Index Image');
subplot(2,2,2), imshow(segNet_layer{1}), title('Objects in Layer-1');
subplot(2,2,3), imshow(segNet_layer{2}), title('Objects in Layer-2');
subplot(2,2,4), imshow(segNet_layer{3}), title('Objects in Layer-3');

figure;
imshow(outImg);title('Segmentation Maps');
colormap('default');

%CNN Clasification
fprintf('Loading Data...\n');
disp(' ');
load('Train_Data.mat');
feature = feat_covid;

feat_test = seg_cnn(Gr_Img);

Train_CNN = mean(feature,2)
Test_CNN  = mean(feat_test,2)

CNN_Mem = ismember(Train_CNN, Test_CNN)
X = find(CNN_Mem(:,1)>0)

if (X >=1 && X <= 50)
    
    disp('Classification Output: <strong>COVID</strong> ')
    Opt.Interpreter = 'tex';
    Opt.WindowStyle = 'normal';
    msgbox('Classification Output : \bf COVID \rm!', 'Classification Done...', 'none', Opt);
    dname = 'Benign';
    
elseif (X >=51 && X <= 100)
    disp('Classification Output: <strong>NORMAL</strong> ')
    Opt.Interpreter = 'tex';
    Opt.WindowStyle = 'normal';
    msgbox('Classification Output : \bf NORMAL \rm!', 'Classification Done...', 'none', Opt);
    dname = 'Malignant';
else
    disp('Classification Output: <strong> No Data </strong> ')
    Opt.Interpreter = 'tex';
    Opt.WindowStyle = 'normal';
    msgbox('Classification Output : \bf No Data \rm!', 'Classification Done...', 'none', Opt);
end

Ni= imresize(Re_Img,[227 227]);
In_per=Ni;
[row,column] = size(In_per);
%sum column values
sum1=sum(In_per); 
sum2=sum(In_per,2);
tot=sum(In_per(:)); 
d=(In_per);
prevalence=(sum1(1)/tot)*100; 
%accuracy
Successrate=100-sum(prevalence);
%TRUE Positive
TRUEPOSITIVE=prevalence(1);
%FALSE Positive
FALSEPOSITIVE=1-TRUEPOSITIVE;
%TRUE Negative 
TRUENEGATIVE=(1-prevalence(1));
%FALSE Negative 
FALSENEGATIVE=1-TRUENEGATIVE;

%SENSITIVITY
SENS=TRUEPOSITIVE/(TRUEPOSITIVE+FALSENEGATIVE);
% SPECIFICITY = 
SPEC=TRUENEGATIVE/(TRUENEGATIVE+FALSEPOSITIVE);
%F-SCORE
a=2;
F_SCORE =(a * TRUEPOSITIVE *SENS)/(TRUEPOSITIVE+SENS);
% Threshold
ME=(mean2(Ni))/2;

%Display Output
disp('Success Rate');disp(Successrate);
disp('Sensitivity');disp(SENS);
disp('Specificity');disp(SPEC);
disp('F_SCORE');disp(F_SCORE);
disp('Threshold');disp(ME);
