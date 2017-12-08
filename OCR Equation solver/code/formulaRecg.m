% This function is the hook-up of all the process of the image
% and get the computation result
% This line is used when applied on an Android phone
% function  formulaRecg(input_img_path ,output_img_path)
% This line is used when applied in Matlan
function  formulaRecg(InputImg)

% Entrance of the application
%-----------------------------------------------------------------------------
% processing function for formula recognition
%------------------------------------------------------------------------------
% INPUT:
% input_img_path 	- input image path
% output_img_path 	- output image path
%--------------------------------------------------------------------------
% This line should be uncommented if applied on a phone
% InputImg = imread(input_img_path);
% preprocess the image
candidates = preprocess(InputImg);

% send to ocr
for i = 1:length(candidates)
    tmp = candidates{i};
    [results{i} b] = OCR(tmp);
end

% postprocess the results, to get only one candidate
result = postprocess(results);

% check the result to see whether it is a valid formula
% only with numbers and operators + - * /
% and the first character and last character should be numbers
if (check(result)==1)
    % calculate the result
    final = trycal(result);
    % create a new canvas to return a image
    height = size(InputImg, 1);
    width = size(InputImg, 2);

    newpic = ones(height, width)*255;

    figure, imshow(newpic)
	% write text to the new canvas
    text(200, 200, result, 'FontSize', 40);
    text(200, 360, '=', 'FontSize', 40);
    text(300, 360, num2str(final), 'FontSize', 40);
    saveas(gcf, 'newpic.jpg');
    imgBinaryLocal = imread('newpic.jpg');
else
    height = size(InputImg, 1);
    width = size(InputImg, 2);

    newpic = ones(height, width)*255;

    figure, imshow(newpic)
    text(200, 200, result, 'FontSize', 40);
    text(200, 360, 'Not recognized', 'FontSize', 40);
    saveas(gcf, 'newpic.jpg');
    imgBinaryLocal = imread('newpic.jpg');
end
% write the result
% This line should be uncommented when applied on a phone
% imwrite(imgBinaryLocal, output_img_path, 'jpg');