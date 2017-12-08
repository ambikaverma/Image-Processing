% This file is for get the formula from OCR
% input: the processed image
% output: the text formula
function [sResults iFileSize] = OCR( I )


% Wrapper function for tesseract
sTempJpegName = '_tesseract_temp.jpg';

% Dump the image to file
% imwrite( I, sTempJpegName, 'Quality',95);
imwrite( I, sTempJpegName);

% Get file size
stat = dir( sTempJpegName);
iFileSize = stat.bytes;
% fprintf('File size %d', stat.bytes);

% Call tesseract
sCmd = sprintf('python .\\external\\ocr.py %s',sTempJpegName);
[iStatus, sResults] = system(sCmd);


% Remove temporary files
sCmd = sprintf('del %s',sTempJpegName);
system(sCmd);

end

