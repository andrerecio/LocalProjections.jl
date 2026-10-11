function r = mvnrnd(mu, Sigma, cases)
% Octave shim for MATLAB's mvnrnd(mu, Sigma, cases) (Statistics Toolbox),
% restricted to the call pattern of the Gorodnichenko-Lee code: one mean
% vector, one covariance matrix, `cases` draws. As in MATLAB, a column mu
% whose length matches Sigma is interpreted as a row vector, the factor is
% cholcov(Sigma), and r = randn(cases, size(T,1)) * T + mu.
[n, d] = size(mu);
if d == 1 && size(Sigma, 1) == n
    mu = mu';
    [n, d] = size(mu);
end
if nargin < 3
    cases = n;
end
[T, err] = cholcov(Sigma);
if err ~= 0
    error('mvnrnd:BadCovariance', 'SIGMA must be a symmetric positive semi-definite matrix.');
end
r = randn(cases, size(T, 1)) * T + repmat(mu, cases, 1);
end
