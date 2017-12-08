% This file is for debluring a image
% input: the original image
% output: the clear image after dilation and erosion
function [imgOut] = deblur(img)
% Perform iterative enhancement
numIterations = 10;
SE = strel('disk', 10);
for nIter = 1:numIterations
    imgDilate = imdilate(img, SE);
    imgErode = imerode(img, SE);
    imgMid = 0.5*(imgDilate + imgErode);
    idxDilate = find(img >= imgMid);
    idxErode = find(img < imgMid);
    img(idxDilate) = imgDilate(idxDilate);
    img(idxErode) = imgErode(idxErode);
end % nIter
imgOut = img;
end