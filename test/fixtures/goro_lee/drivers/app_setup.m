function [root, ref, outdir] = app_setup()
% Put the read-only Gorodnichenko-Lee reference tree and the Octave shims
% on the path. Returns the repo root, the reference tree and the fixture
% output directory (test/fixtures/goro_lee).
here   = fileparts(mfilename('fullpath'));
outdir = canonicalize_file_name(fullfile(here, '..'));
root   = canonicalize_file_name(fullfile(here, '..', '..', '..', '..'));
ref    = fullfile(root, 'docs', 'src', 'code', 'goro_lee_r2');
if ~exist(fullfile(ref, 'lib', 'SubR2.m'), 'file')
    error('Reference tree not found: %s (gitignored; restore it first)', ref);
end
addpath(fullfile(outdir, 'octave_shims'));
addpath(ref);
addpath(fullfile(ref, 'lib'));
end
