function [p, s] = dqb_fit_predict(model, Xtr, ytr, Xte, feat, seed)
%DQB_FIT_PREDICT  Train one model on (Xtr, ytr) and predict Xte.
%   model: 'Boosted trees', 'Random Forest', 'SVR', 'GPR', 'Ridge', 'Constant'.
%   All scaling and transformations are fitted on the training part only.
%   s returns the GPR predictive standard deviation (NaN otherwise).
%   Requires the Statistics and Machine Learning Toolbox for all models
%   except Ridge and Constant.

s = nan(size(Xte, 1), 1);
ytr = ytr(:);
switch model
    case 'Constant'
        p = repmat(mean(ytr), size(Xte, 1), 1);

    case 'Ridge'
        [A, B] = local_prep(Xtr, Xte, feat, true);
        p = local_ridgecv(A, ytr, B, logspace(-3, 3, 25));

    case 'Random Forest'           % 500 fully grown trees, all descriptors at each split
        rng(seed, 'twister');
        m = TreeBagger(500, Xtr, ytr, 'Method', 'regression', ...
            'NumPredictorsToSample', size(Xtr, 2), 'MinLeafSize', 1);
        p = predict(m, Xte);

    case 'Boosted trees'           % least-squares boosting of shallow trees (depth ~3)
        rng(seed, 'twister');
        t = templateTree('MaxNumSplits', 7);
        m = fitrensemble(Xtr, ytr, 'Method', 'LSBoost', 'NumLearningCycles', 500, ...
            'LearnRate', 0.1, 'Learners', t);
        p = predict(m, Xte);

    case 'SVR'                     % RBF kernel, C / gamma / epsilon by inner 5-fold grid search
        [A, B] = local_prep(Xtr, Xte, feat, true);
        P = size(A, 2);
        Cs = [0.3 1 3 10 30 100];
        gs = [1 / P, 0.003, 0.01, 0.03, 0.1];          % first value = sklearn 'scale' for standardised data
        es = [0.01 0.05 0.1];
        nI = size(A, 1); pf = dqb_perm(nI, 1000 + seed);
        fo = zeros(nI, 1); fo(pf) = mod(0:nI - 1, 5) + 1;
        best = inf; arg = [1 1 1];
        for ic = 1:numel(Cs)
            for ig = 1:numel(gs)
                for ie = 1:numel(es)
                    e = 0;
                    for k = 1:5
                        tr = fo ~= k;
                        m = fitrsvm(A(tr, :), ytr(tr), 'KernelFunction', 'gaussian', ...
                            'KernelScale', 1 / sqrt(gs(ig)), 'BoxConstraint', Cs(ic), ...
                            'Epsilon', es(ie), 'Standardize', false);
                        e = e + sum((predict(m, A(~tr, :)) - ytr(~tr)).^2);
                    end
                    if e < best, best = e; arg = [ic ig ie]; end
                end
            end
        end
        m = fitrsvm(A, ytr, 'KernelFunction', 'gaussian', 'KernelScale', 1 / sqrt(gs(arg(2))), ...
            'BoxConstraint', Cs(arg(1)), 'Epsilon', es(arg(3)), 'Standardize', false);
        p = predict(m, B);

    case 'GPR'                     % Matern 5/2 with one length scale per descriptor (ARD)
        [A, B] = local_prep(Xtr, Xte, feat, true);
        rng(seed, 'twister');
        m = fitrgp(A, ytr, 'KernelFunction', 'ardmatern52', 'BasisFunction', 'constant', ...
            'Standardize', false, 'FitMethod', 'exact', 'PredictMethod', 'exact');
        [p, s] = predict(m, B);

    otherwise
        error('dqb:model', 'Unknown model "%s".', model);
end
p = p(:);
end

% ---------------------------------------------------------------------
function [A, B] = local_prep(Xtr, Xte, feat, uselog)
% log10(1/R^2) for distance/kernel/linear models, then standardisation
% with the mean and SD of the training part.
j = find(strcmpi(feat, '1/r2') | strcmpi(feat, '1/R2'));
if uselog && ~isempty(j)
    Xtr(:, j) = log10(Xtr(:, j)); Xte(:, j) = log10(Xte(:, j));
end
mu = mean(Xtr, 1); sd = std(Xtr, 0, 1); sd(sd == 0) = 1;
A = (Xtr - mu) ./ sd; B = (Xte - mu) ./ sd;
end

function p = local_ridgecv(A, y, B, alphas)
% Ridge regression with intercept; alpha chosen by exact leave-one-out error.
n = size(A, 1);
xm = mean(A, 1); ym = mean(y);
Ac = A - xm; yc = y - ym;
[U, S, V] = svd(Ac, 'econ'); d = diag(S);
best = inf; wbest = zeros(size(A, 2), 1);
for a = alphas
    f = d.^2 ./ (d.^2 + a);
    h = sum((U.^2) .* f', 2) + 1 / n;        % leverage incl. intercept
    r = yc - U * (f .* (U' * yc));
    loo = mean((r ./ (1 - h)).^2);
    if loo < best
        best = loo; wbest = V * ((d ./ (d.^2 + a)) .* (U' * yc));
    end
end
p = (B - xm) * wbest + ym;
end
