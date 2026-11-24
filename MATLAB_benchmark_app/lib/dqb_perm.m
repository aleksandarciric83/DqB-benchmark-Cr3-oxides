function p = dqb_perm(n, seed)
%DQB_PERM  Reproducible random permutation of 1:n for a given seed.
if exist('RandStream', 'class') == 8 || exist('RandStream', 'file') == 2
    s = RandStream('mt19937ar', 'Seed', seed);
    p = randperm(s, n);
else                                   % Octave
    rand('twister', seed); %#ok<RAND>
    [~, p] = sort(rand(1, n));
end
p = p(:);
end
