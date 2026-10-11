function syn_writecsv(fname, M)
% Write a numeric (or logical) matrix as headerless CSV with '%.17g' (round-trip
% exact for Float64). NaN/Inf are written as NaN/Inf. Empty matrices are an error.
M = double(M);
if isempty(M)
    error('syn_writecsv: refusing to write an empty matrix to %s', fname);
end
[~, n] = size(M);
fmt = [repmat('%.17g,', 1, n - 1), '%.17g\n'];
fid = fopen(fname, 'w');
if fid < 0
    error('syn_writecsv: cannot open %s', fname);
end
fprintf(fid, fmt, M.');
fclose(fid);
end
