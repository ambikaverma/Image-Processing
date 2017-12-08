% This file is for locating the formula in the original image
% input: the original image, the image after some process
% output: a patch image
function result = boundingbox(image, image_bw)

[row, col] = size(image);

row_min = row;
row_max = 0;
col_min = col;
col_max = 0;
% find the most left, right, top, bottom point
for i = 1:row
    for j = 1:col
        if image(i, j) == 1
            if i >= row_max               
                row_max = i;
            end
            if i <= row_min                
                row_min = i;
            end
            if j >= col_max                
                col_max = j;
            end
            if j <= col_min
                col_min = j;
            end
        end
    end
end

result = image_bw(row_min:row_max, col_min:col_max);