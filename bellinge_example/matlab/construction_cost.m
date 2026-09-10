function CB = construction_cost(mian_ji)
    zhsqy_number = length(mian_ji) ./ 3;
    CB_before = reshape(mian_ji .* 10000, 3, zhsqy_number);
    CB = sum([600 500 150] * CB_before);
end
