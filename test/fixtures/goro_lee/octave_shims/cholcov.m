function [T, p] = cholcov(Sigma, flag)
% Octave shim for MATLAB's cholcov (Statistics Toolbox), following the
% MATLAB algorithm: Cholesky factor if Sigma is (numerically) symmetric
% positive definite; otherwise, if flag ~= 0, an eigenvalue-based factor
% T with Sigma = T'*T when Sigma is positive semi-definite (p = number of
% negative eigenvalues, T = [] if p > 0); p = NaN if Sigma is not square
% and symmetric.
if nargin < 2
    flag = 1;
end
[n, m] = size(Sigma);
tol = 10 * eps(max(abs(diag(Sigma))));
if (n == m) && all(all(abs(Sigma - Sigma') < n * tol))
    [T, p] = chol(Sigma);
    if p > 0
        if flag
            [U, D] = eig(full((Sigma + Sigma') / 2));
            % Pick eigenvector direction so max abs coordinate is positive
            [~, maxind] = max(abs(U), [], 1);
            negloc = (U(maxind + (0:n:(m-1)*n)) < 0);
            U(:, negloc) = -U(:, negloc);
            D = diag(D);
            tol = eps(max(D)) * length(D);
            t = (abs(D) > tol);
            D = D(t);
            p = sum(D < 0);
            if p == 0
                T = diag(sqrt(D)) * U(:, t)';
            else
                T = zeros(0, class(Sigma));
            end
        else
            T = zeros(0, class(Sigma));
        end
    end
else
    T = zeros(0, class(Sigma));
    p = NaN;
end
end
