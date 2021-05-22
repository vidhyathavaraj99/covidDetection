function [ he, K1, K2, K3 ] = features_data(Img)
%% % % % % % % % % K-MEANS CLUSTRING% % % % % % % % % % % % 
% he=imresize(Img, [512 512]);
he=Img;
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
K1 = segmented_images{1};
K2 = segmented_images{2};
K3 = segmented_images{3};
% figure;
% subplot(1,3,1); imshow(segmented_images{1}), title('objects in cluster 1');
% subplot(1,3,2); imshow(segmented_images{2}), title('objects in cluster 2');
% subplot(1,3,3); imshow(segmented_images{3}), title('objects in cluster 3');


end

