% This file is used to adjust the oblique of the image
% input: the image after rotation and boundingbox.m

function result = oblique(binary)

% find the degree of oblique
tl = zeros(2,1);
tr = zeros(2,1);
bl = zeros(2,1);
br = zeros(2,1);

[row, col] = size(binary);

    % find the top left pixel
    label = false;
    for i = 1:1:row
        for j = 1:1:col
            pixel = binary(i, j);
            if pixel == 1
                tl = [i, j];
                break;
            end
        end
        
        for k = col:-1:1
            pixel = binary(i, k);
            if pixel == 1
                tr = [i, k];
                break;
            end
        end
        
        if tr(2) - tl(2) < 0.8*col
            continue;
        else
            break;
        end

    end

    % find the bottom left pixel
    label = false;
    for i = row:-1:1
        for j = 1:1:col
            pixel = binary(i, j);
            if pixel == 1
                if j < tl(2)
                    bl = [i, j];
                    label = true;
                    break;
                else
                    break;
                end                
            end            
        end
        
        if label == true
            break;
        end
    end
    
    % calculate the degree of oblique
    up_line = tr(2) - tl(2);
    delta_half = tl(2) - bl(2);
    bottom_line = 2*delta_half + up_line;
    oblique_rate = 4*up_line/bottom_line

    if oblique_rate <= 0
        oblique_rate = 4;
    end


    % Transform to a quadrilateral with vertices
    udata = [0 1];  vdata = [0 1];
    tform = maketform('projective',[0 0; 1 0; 1 1; 0 1], [-4 -4; 4 -4; oblique_rate 4; -1*oblique_rate 4]);


    % Fill with gray and use bicubic interpolation. 
    % Make the output size the same as the input size.
    temp = imtransform(binary, tform, 'bicubic', 'udata', udata, 'vdata', vdata, 'size', size(binary), 'fill', 128);    
	temp = double(temp);
    [temp_row, temp_col] = size(temp);

	% make the background to be black
	for i=1:temp_row
		for j = 1:temp_col
			if temp(i,j)>1
				temp(i,j) = 0;
			end
		end
	end

	% remove small white dots on the edge
	imgLabel = bwlabel(temp);
	shapeProps = regionprops(imgLabel, 'Area');
	for nRegion = 1:length(shapeProps)
		idx = find(imgLabel == nRegion);
		area = shapeProps(nRegion).Area;
		if area < 10
			temp(idx) = 0;
		end
	end
	
result = temp;
% figure, imshow(result);