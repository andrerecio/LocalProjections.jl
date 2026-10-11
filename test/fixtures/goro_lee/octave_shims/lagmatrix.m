function YLag = lagmatrix(Y, lags)
% Octave shim for MATLAB's lagmatrix (Econometrics Toolbox), MATLAB semantics:
%   - Y is T-by-N (a row vector is treated as a single series);
%   - a positive lag L shifts down: YLag(t) = Y(t-L), first L rows NaN;
%   - a negative lag -L is a lead:  YLag(t) = Y(t+L), last L rows NaN;
%   - lag 0 returns Y;
%   - the output has N*numel(lags) columns grouped by lag, in the order
%     given: [Y(t-lags(1)), Y(t-lags(2)), ...], each block N columns wide.
% Written for the LocalProjections.jl Gorodnichenko-Lee fixtures.
if isrow(Y)
    Y = Y(:);
end
[T, N] = size(Y);
lags = lags(:)';
YLag = nan(T, N * numel(lags));
for j = 1:numel(lags)
    L = lags(j);
    cols = (j-1)*N + (1:N);
    if L >= 0
        if L < T
            YLag(L+1:T, cols) = Y(1:T-L, :);
        end
    else
        L = -L;
        if L < T
            YLag(1:T-L, cols) = Y(L+1:T, :);
        end
    end
end
end
