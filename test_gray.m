clc;
close all;
clear all;

% Read files form pc. 
[file,path,indx] = uigetfile('./Input Image/non_covid/*.jpg;*.jpeg;*.bmp',... 
                                    'Select an Input Image File');
In_Img = imread([path,file]);


figure; imshow(In_Img); title('Input Test Image');

% Get the dimensions of the image.  
[rows, columns, no_of_band] = size(In_Img);
if isequal (no_of_band,3)
	% Convert it to gray scale 
    Gr_Img = imresize(In_Img,[256 256]);
    figure; imshow(Gr_Img); title('Band 3')
else
    In_Im(:,:,1) = In_Img;
    In_Im(:,:,2) = In_Img;
    In_Im(:,:,3) = In_Img;
    
    Img = imresize(In_Im,[256 256]);
    figure; imshow(Img); title('Band Corrected')
end