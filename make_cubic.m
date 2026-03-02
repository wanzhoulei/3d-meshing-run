function make_tetflip_dataset(outFile, N, sigma, seed)
%MAKE_TETFLIP_DATASET  Synthetic tetra flip dataset (grid + interior perturb).
%
% outFile : e.g. 'tetflip_dataset.mat'   (saved with -v7.3 for Python/h5py)
% N       : number of samples
% sigma   : std dev of Gaussian perturbation on interior points (e.g. 1e-2)
% seed    : RNG seed for reproducibility
%
% Requires DistMesh functions on path for simpqual (and optionally simpplot).

    if nargin < 1, outFile = 'tetflip_dataset.mat'; end
    if nargin < 2, N = 2000; end
    if nargin < 3, sigma = 1e-2; end
    if nargin < 4, seed = 42; end

    rng(seed);

    % --- base 5x5x5 grid in [0,1]^3 => 125 points
    s = 0:0.25:1;
    [x,y,z] = ndgrid(s,s,s);
    P0 = [x(:), y(:), z(:)];   % 125x3

    % interior points are those strictly inside the cube => 3^3 = 27
    interiorMask = (P0(:,1) > 0 & P0(:,1) < 1 & ...
                    P0(:,2) > 0 & P0(:,2) < 1 & ...
                    P0(:,3) > 0 & P0(:,3) < 1);
    interiorIdx = find(interiorMask);
    assert(numel(interiorIdx) == 27, 'Expected 27 interior points.');

    % --- reference topology: Delaunay on unperturbed points
    % Note: grid points are degenerate for Delaunay; qhull picks one.
    % This is fine for a toy "oracle-ish" connectivity.
    T_ref = delaunayn(P0);   % M0x4

    % Prepare struct array
    samples = repmat(struct( ...
        'P', [], ...
        'T_init', [], ...
        'T_ref', [], ...
        'sigma', [], ...
        'minq_init', [], ...
        'minq_ref', []), N, 1);

    for i = 1:N
        % Perturb only interior points
        P = P0;
        P(interiorIdx,:) = P(interiorIdx,:) + sigma * randn(numel(interiorIdx), 3);

        % Initial (often bad) triangulation: Delaunay on perturbed points
        T_init = delaunayn(P);

        % Quality measures
        % (simpqual returns per-tet quality; we keep only mins here)
        q_init = simpqual(P, T_init);
        q_ref  = simpqual(P, T_ref);

        samples(i).P         = P;                 % double is fine
        samples(i).T_init    = int32(T_init);
        samples(i).T_ref     = int32(T_ref);
        samples(i).sigma     = sigma;
        samples(i).minq_init = min(q_init);
        samples(i).minq_ref  = min(q_ref);
    end

    meta = struct();
    meta.s = s;
    meta.P0 = P0;
    meta.interiorIdx = int32(interiorIdx);
    meta.T_ref = int32(T_ref);
    meta.N = N;
    meta.sigma = sigma;
    meta.seed = seed;

    save(outFile, 'samples', 'meta', '-v7.3');
    fprintf('Saved %d samples to %s\n', N, outFile);
end
