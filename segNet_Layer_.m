function [segmented_images,pixel_labels, class_im] = segNet_Layer_(Img)
% he=imresize(Img, [512 512]);
he=Img;

% Get the dimensions of the image.  
[rows, columns, no_of_band] = size(Img);
if isequal (no_of_band,3)
	% Convert it to gray scale 
    he=Img;
else
    In_Im(:,:,1) = Img;
    In_Im(:,:,2) = Img;
    In_Im(:,:,3) = Img;
    he = In_Im;
end

% Convert Image Color Space
cform = makecform('srgb2lab');
lab_he = applycform(he,cform);

% Classify Space Using K-Means Clustering
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);

nColors = 3;

% repeat the clustering 3 times to avoid local minima
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
                                  
                                  
% Label Every Pixel in the Image Using the Results from KMEANS
pixel_labels = reshape(cluster_idx,nrows,ncols);
% figure; imshow(pixel_labels,[]), title('image labeled by cluster index');

% Create Images that Segment the H&E Image by Color.
segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels,[1 1 3]);

for k = 1:nColors
    color = he;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color;
end

[rows, columns, no_of_band] = size(he);
if isequal (no_of_band,3)
	% Convert it to gray scale 
	class_gray = rgb2gray(he);
    class_im = imresize(class_gray, [256 256]);
else
    class_im = imresize(he,[256 256]);
end

end

