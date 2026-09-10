function data = readNPY(filename)
[shape, dataType, fortranOrder, littleEndian, totalHeaderLength, ~] = readNPYheader(filename);
if littleEndian
    fid = fopen(filename, 'r', 'l');
else
    fid = fopen(filename, 'r', 'b');
end
try
    [~] = fread(fid, totalHeaderLength, 'uint8');
    data = fread(fid, prod(shape), [dataType '=>' dataType]);
    if length(shape)>1 && ~fortranOrder
        data = reshape(data, shape(end:-1:1));
        data = permute(data, [length(shape):-1:1]);
    elseif length(shape)>1
        data = reshape(data, shape);
    end
    fclose(fid);
catch me
    fclose(fid);
    rethrow(me);
end
