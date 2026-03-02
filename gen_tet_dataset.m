% gen_tet_dataset.m
% Generate grid-125 cube dataset with interior perturbations
% Saves a .mat that Python can load via scipy.io.loadmat

clear; clc;

% ---------------- user params ----------------
Nsamples   = 2000;
sigma      = 1e-2;     % absolute perturbation magnitude
seed       = 0;
out_file   = sprintf('tet_dataset_grid125_sigma%.0e_N%d.mat', sigma, Nsamples);

s = 0:0.25:1;
[x,y,z] = ndgrid(s,s,s);
p0 = [x(:), y(:), z(:)];      % (125,3) unperturbed points
T_good = delaunayn(p0);       % constant "good topology"

% which points are "interior block" (your 3x3x3)
ix = 2:4; iy = 2:4; iz = 2:4;

rng(seed);

P      = cell(Nsamples,1);
T_bad  = cell(Nsamples,1);
minQ_bad  = zeros(Nsamples,1);
minQ_good = zeros(Nsamples,1);

for n = 1:Nsamples
    x1 = x; y1 = y; z1 = z;

    x1(ix,iy,iz) = x1(ix,iy,iz) + sigma * randn(numel(ix),numel(iy),numel(iz));
    y1(ix,iy,iz) = y1(ix,iy,iz) + sigma * randn(numel(ix),numel(iy),numel(iz));
    z1(ix,iy,iz) = z1(ix,iy,iz) + sigma * randn(numel(ix),numel(iy),numel(iz));

    p1 = [x1(:), y1(:), z1(:)];

    % "bad" mesh from Delaunay on perturbed points
    t1 = delaunayn(p1);

    % qualities computed on the SAME coordinates p1
    q_bad  = simpqual(p1, t1);
    q_good = simpqual(p1, T_good);

    P{n} = p1;
    T_bad{n} = t1;

    minQ_bad(n)  = min(q_bad);
    minQ_good(n) = min(q_good);

    if mod(n,200) == 0
        fprintf('n=%d: minQ_bad=%.4g, minQ_good=%.4g\n', n, minQ_bad(n), minQ_good(n));
    end
end

params = struct();
params.Nsamples = Nsamples;
params.sigma = sigma;
params.seed = seed;
params.grid_s = s;
params.interior_ix = ix;
params.interior_iy = iy;
params.interior_iz = iz;

save(out_file, 'P', 'T_bad', 'T_good', 'minQ_bad', 'minQ_good', 'params', '-v7.3');
fprintf('Saved dataset to %s\n', out_file);
