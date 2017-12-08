% This file is for calculating the result of the formula
% input: string of formula
% output: the number

function final = trycal(input)
    N = length(input);
    idxnum = 1;
    idxop = 1;
    i = 1;
    while(i <= N)
        if (isspace(input(i)))
            break;
        end
        if (isempty(str2num(input(i))) == 0)
            number = str2num(input(i));
            while((i+1 <= N) && isempty(str2num(input(i+1))) == 0)
                number = number*10 + str2num(input(i+1));
                i = i+1;
            end
            Numvector(idxnum) = number;
            idxnum = idxnum + 1;
        else
            OperatorVector(idxop) = input(i);
            idxop = idxop + 1;
        end
        i = i + 1;
    end
    Nop = length(OperatorVector);
    result = '';
    for i = 1:Nop
        result = strcat(result, 'x(', num2str(i), ')', OperatorVector(i));
    end
    result = strcat(result,  'x(', num2str(Nop+1), ')');
    f = inline(result);
    if (length(Numvector) == length(OperatorVector)+1)
        final = f(Numvector);
    else
        final = 'not';
    end
end
