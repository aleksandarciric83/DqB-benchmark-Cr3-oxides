function test = dqb_splits(D, scheme, K, seed)
%DQB_SPLITS  Test-index sets for one partition of the data.
%   scheme: 'compositions' - K-fold, all entries of a formula in one fold
%           'spacegroups'  - K-fold grouped by space group number
%           'sites'        - leave-one-group-out by substituted cation
%           'kfold'        - ungrouped K-fold (as in Kumar et al.)
%           'transfer'     - train on inherited entries, test on added hosts
%   Returns a cell array of logical test masks (training = complement).

n = D.n;
switch scheme
    case 'compositions'
        [~, ~, g] = unique(lower(D.formula));
        test = local_groupkfold(g, K, seed);
    case 'spacegroups'
        [~, ~, g] = unique(D.sg);
        test = local_groupkfold(g, K, seed);
    case 'sites'
        [u, ~, g] = unique(D.site);
        test = cell(1, numel(u));
        for k = 1:numel(u), test{k} = (g == k); end
    case 'kfold'
        p = dqb_perm(n, seed);
        f = zeros(n, 1); f(p) = mod(0:n - 1, K) + 1;
        test = cell(1, K);
        for k = 1:K, test{k} = (f == k); end
    case 'transfer'
        inh = ismember(lower(D.subset), {'kumar', 'inherited', 'kumar et al. (curated)'});
        if ~any(inh) || all(inh)
            error('dqb:split', 'Transfer needs a Subset column with inherited (Kumar) and added entries.');
        end
        test = {~inh};
    otherwise
        error('dqb:split', 'Unknown scheme "%s".', scheme);
end
end

function test = local_groupkfold(g, K, seed)
% Groups are shuffled, then assigned largest-first to the currently
% smallest fold (as sklearn GroupKFold with shuffle).
G = max(g);
K = min(K, G);
sz = accumarray(g(:), 1, [G 1]);
order = dqb_perm(G, seed);
[~, s] = sort(sz(order), 'descend');      % stable sort keeps the random order for ties
order = order(s);
load_ = zeros(K, 1); fold = zeros(G, 1);
for i = 1:G
    [~, k] = min(load_);
    fold(order(i)) = k;
    load_(k) = load_(k) + sz(order(i));
end
test = cell(1, K);
for k = 1:K, test{k} = (fold(g) == k); end
end
