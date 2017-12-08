% This file is for preprocessing function for formula recognition
function images_out = preprocess(InputImg)
%------------------------------------------------------------------------------
% INPUT:
% input_img_path 	- input image path
% output_img_path 	- output image path
%--------------------------------------------------------------------------

InputImg = imread('E:\EE 371 R\OCR\test.jpg');
figure
imshow(InputImg);
title('Original Image');
InputImg = im2double(InputImg);
GrayImg = rgb2gray(InputImg);
figure
imshow(GrayImg);
title('Grayscale Image');

% Denoise
[img, imgLP, imgHPSC] = denoiseSoftCoring(GrayImg, 5, 1.75, 30, 2.2);
figure
imshow(img);
% Debur
img = deblur(img);
figure
imshow(img);
% Binarization
winSize = 32;
stepSize = 16;
%imgBW = localAdaptiveOtsu(GrayImg, 32, 16);
imgBW = binarization(GrayImg);
imgBW_temp = imgBW;
% Negate the image to make the backgroud black 
%imgBW = not(imgBW);
figure;
imshow(imgBW);
title('Binarized image');

% localize the patch that's is possibly a formula
patch_list = localization(imgBW);
%localization(imgBW);

% apply bounding box to the patches
candidates = {};
for i = 1:length(patch_list)
    candidates{i} = boundingbox(patch_list{i}, imgBW_temp);
     figure;
     imshow(candidates{i});
     title(['candidate ', num2str(i)]);
end

% Rotate the image
candidates_r = {}
for i = 1:length(candidates)
    candidates_r{i} = rotate(candidates{i});
    figure;
    imshow(candidates_r{i});
    title(['candidate_r ', num2str(i)]);
end

% process oblique image
images_out = {}
for i = 1:length(candidates_r)
    images_out{i} = oblique(candidates_r{i});
    images_out{i} = not(images_out{i});
    figure;
    imshow(images_out{i});
    title(['images_out  ', num2str(i)]);
end

end

