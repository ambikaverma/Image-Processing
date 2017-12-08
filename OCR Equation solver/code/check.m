% This file is for checking whether the string is a formula
% input: a string
% output: is a formula or not
function isformula = check(string)
    N = length(string);
    flag = 1;
	% check the first character
    if (string(1) == '+' || string(1) == '-' || string(1) == '*' || string(1) == '/')
        flag = 0;
	% check the last character
    elseif (string(N) == '+' || string(N) == '-' || string(N) == '*' || string(N) == '/')
        flag = 0;
	% check every character
    else
        for i = 1:N
            if(isempty(str2num(string(i))) == 0)
                continue;
            elseif (string(i)=='+')
                continue;
            elseif (string(i)=='-')
                continue;
            elseif (string(i)=='*')
                continue;
            elseif (string(i)=='/')
                continue;
            else
                flag = 0;
                break;
            end
        end
    end
    isformula = flag;
end