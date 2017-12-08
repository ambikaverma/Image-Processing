% This file is for finding the location of the formula

function patch_list  = localization(imgBW) 

imgLabel = bwlabel(imgBW);
% First, filter out clutters according to area
shapeProps = regionprops(imgLabel, 'Area');
allArea = [shapeProps.Area];
minArea = min(allArea);
% get rid of extremely small regions
small = find([shapeProps.Area] < 20)
for i = 1:length(small)
    idx = find(imgLabel == small(1,i));
    imgBW(idx) = 0;
end
% figure;
% imshow(imgBW);
% title('small area removed');
imgLabel = bwlabel(imgBW);
shapeProps = regionprops(imgLabel, 'Area');
allArea = [shapeProps.Area];
minArea = min(allArea);
maxArea = max(allArea);
if (maxArea - minArea)/minArea > 10 

    % find statistics of labels' areas
    nBin = 20;
    step_size = (maxArea - minArea)/nBin;
    n = 1;
    areaBin = zeros(1, nBin+10);

    area_i = 0;
    while(1)
        for nRegion = 1:length(shapeProps)
            area = shapeProps(nRegion).Area;
            if( area>= area_i && area < area_i + step_size)
                areaBin(n) = areaBin(n) + 1;
            end
        end
        n = n+1;
        area_i = area_i + step_size;
        if area_i > maxArea break; end
    end

    % get rid of area range containing small number of labels
    target_idx = find(areaBin == 1);
    [ h n_target] = size(target_idx)
    for j = 1:n_target
        if areaBin(target_idx(j)) == 0 continue; end
        upper = minArea + target_idx(j)*step_size + 0.2*step_size;
        lower = minArea + (target_idx(j) - 1)*step_size - 0.2*step_size;
        for nRegion = 1:length(shapeProps)
            idx = find(imgLabel == nRegion);
            area = shapeProps(nRegion).Area;
            if (area <= upper ) && (area > lower)
                imgBW(idx) = 0;
            end
%             figure;
%             imshow(imgBW);
%             title('me');
        end
    end

    figure;
    imshow(imgBW);
    title('Filtered based on area');
end

% % find patches according to major axis length
imgLabel = bwlabel(imgBW);
shapeProps = regionprops(imgLabel, 'MajorAxisLength');
allLength = [shapeProps.MajorAxisLength];
minLength = min(allLength);
maxLength = max(allLength);
% Get statistics about labels' axis length
step_size = (maxLength - minLength)/2;
lengthBin = zeros(1,20);
n = 1;
length_i = minLength;
while(1)
    for nRegion = 1:length(shapeProps)
        axisLength = shapeProps(nRegion).MajorAxisLength;
        if( axisLength>= length_i && axisLength < length_i + step_size)
            lengthBin(n) = lengthBin(n) + 1;
        end
    end
    n = n+1;
    length_i = length_i + step_size;
    if length_i > maxLength break; end
end

% find candidate equation patch
target_idx = find(lengthBin > 2);
[h n_target] = size(target_idx);
for i = 1:n_target
    imgBW_copy = imgBW;
    upper = minLength + target_idx(i)*step_size;
    lower = minLength + (target_idx(i)-1)*step_size;
    for nRegion = 1:length(shapeProps)
        idx = find(imgLabel == nRegion);
        axisLength = shapeProps(nRegion).MajorAxisLength;
        if (axisLength > upper ) || (axisLength < lower)
            imgBW_copy(idx) = 0;
        end
    end
     patch_list{i} = imgBW_copy;
     figure;
     imshow(patch_list{i});
     title(['patch ', num2str(i)]);
end

end

