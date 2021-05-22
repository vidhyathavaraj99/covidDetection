function annlayer = seg_cnn(image)
grayImage = imageToGray(image);
ctImage = segNet_cnn_layer(grayImage);
annlayer = imhist(ctImage)';
% drop first and last column in histogram
annlayer = annlayer(2:end-1);
end
