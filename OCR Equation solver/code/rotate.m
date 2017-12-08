% This file rotate a slew image to horizontal
% input: after binarization, logical
% output: logical
function newImage = rotate(image)
    imagelabel = bwlabel(image);
    shapeProps = regionprops(imagelabel, 'Centroid');
    n = length(shapeProps);
    center = zeros(2, n);
    for nRegion = 1:n
        center(:, nRegion) = shapeProps(nRegion).Centroid;
    end
    if (center(2, n) < center(2, 1))
        center1 = center(: ,1);
        center2 = center(:, n);
    else
%         [p q] = find(center(2, :)==min(center(2, :)));
        center1 = center(:, 1);
        center2 = center(:, n);
    end

    if (abs(center1(2)-center2(2)) <= 2)
        % do nothing
        newImage = image;
    else 
        anglerad = atan((center2(2)-center1(2))/(center1(1)-center2(1)));
        angledeg = 180/pi*anglerad;
        newImage = imrotate(image, -angledeg, 'bilinear');
    end
%     figure
%     imshow(newImage)
%     title('the rotated image')
