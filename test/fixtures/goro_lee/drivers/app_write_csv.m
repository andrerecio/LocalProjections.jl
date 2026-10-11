function app_write_csv(path, header, M)
% Write matrix M to path as CSV with a one-line header (cellstr), every
% value printed with %.17g (round-trip exact for Float64; NaN as NaN).
fid = fopen(path, 'w');
if fid < 0
    error('cannot open %s', path);
end
fprintf(fid, '%s\n', strjoin(header, ','));
fmt = [strjoin(repmat({'%.17g'}, 1, size(M, 2)), ','), '\n'];
fprintf(fid, fmt, M');
fclose(fid);
end
