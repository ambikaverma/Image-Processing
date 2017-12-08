% This file is for denoising a image
% input: the image and parameters
% output: the clear image
function [imgOut, imgLP, imgHPSC] = denoiseSoftCoring(img, lpf, m, tau, gamma)
imgDouble = double(img);
g = ones(lpf,lpf)/(lpf*lpf);
imgLP = imfilter(imgDouble, g, 'replicate');
imgHP = imgDouble - imgLP;
imgHPSC = m * imgHP .* (1 - exp(-(abs(imgHP)/tau).^gamma));
imgOut = imgLP + imgHPSC;
imgOut = uint8(max(0,min(255,imgOut)));
imgLP = uint8(imgLP);
end