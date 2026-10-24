function x = norminv(p, mu, sigma)
% Octave shim for MATLAB's norminv (Statistics Toolbox).
if nargin < 2, mu = 0; end
if nargin < 3, sigma = 1; end
x = mu + sigma .* (-sqrt(2) .* erfcinv(2 .* p));
end
