% This file performs postprocess of a cell of OCR outputs
% It selects a patch which is most likely to be an mathamatical equaltion
% Input: A cell contains the outputs of OCR
% Output: A matrix of char which has a highest probability of being an equaltion

function post_result = postprocess(list)

probOfLetter = zeros(1, length(list));

% calculate the probability of occurrence of alphabets in the string
flag = 0;
for i = 1:length(list)    
    candidate = list{i};
    if length(candidate) == 0
        continue;
    end
    
    flag = 1;
    numberOfLetter = 0;
    for j = 1:length(candidate)
        % change O to 0
        if candidate(j) == 'O'
            candidate(j) = '0';
        end
        % change l to 1
        if candidate(j) == 'l'
            candidate(j) = '1';
        end
        
        % change ~ to -
        if candidate(j) == '~'
            candidate(j) = '-';
        end
        
        % change s to 8
        if candidate(j) == 's' || candidate(j) == 'B'
            candidate(j) = '8';
        end
        
        % determine whether each element is a letter or not
        if isletter(candidate(j)) == true
            numberOfLetter = numberOfLetter +1;
        end
    end
    list{i} = candidate;
    probOfLetter(i) = numberOfLetter/length(candidate);
end

% search for the string that has the lowest probability (highest possibility to be a formula)
if flag == 0
    post_result = 'blank';
else
    temp = 1.1;
    number = 0;
    for i = 1:length(list)
        if probOfLetter(i) < temp
            number = i;
            temp = probOfLetter(i);
        elseif(probOfLetter(i) == temp)
            if length(list{i}) > length(list{number})
                number = i;
            end
        end
    end

    result_temp = list{number};

    % remove space in the string
    k = result_temp==' ';
    result_temp(k)=[];

    % remove the last character which is a space
    post_result = result_temp(1:(length(result_temp)-1));
end

end