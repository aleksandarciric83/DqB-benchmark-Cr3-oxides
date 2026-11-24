function R = dqb_summary(R, nboot)
%DQB_SUMMARY  Metrics, paired bootstrap against ridge regression and
%   crystal-field-regime classification for the result struct of DQB_RUN.
%   Regimes: WCF Dq/B < 2.23, ICF 2.23-2.42, SCF > 2.42.

if nargin < 2, nboot = 4000; end
y = R.y(:); M = numel(R.models);
test = any(~isnan(R.oof{1}), 1)';            % entries that were predicted (all, or the added hosts in transfer)
yt = y(test);
reg = @(v) 1 + (v >= 2.23) + (v > 2.42);     % 1 WCF, 2 ICF, 3 SCF

[~, ~, grp] = unique(lower(R.D.formula(test)));
iR = find(strcmp(R.models, 'Ridge'), 1);

head = {'Model', 'R2', 'R2 SD', 'MAE', 'MAE SD', 'RMSE', 'dMAE vs ridge', '95% CI', 'vs ridge', ...
        'Regime acc.', 'ICF correct'};
rows = cell(M, numel(head));
R.metrics = cell(1, M); R.mean = cell(1, M);
for m = 1:M
    O = R.oof{m}(:, test);
    k = size(O, 1); v = zeros(k, 3);
    for r = 1:k, q = dqb_metrics(yt, O(r, :)'); v(r, :) = [q.R2 q.MAE q.RMSE]; end
    R.metrics{m} = v;
    pm = mean(O, 1)'; R.mean{m} = pm;
    acc = mean(reg(pm) == reg(yt));
    icf = reg(yt) == 2;
    rows(m, [1:6 10 11]) = {R.models{m}, mean(v(:, 1)), local_sd(v(:, 1)), mean(v(:, 2)), local_sd(v(:, 2)), ...
        mean(v(:, 3)), sprintf('%.0f %%', 100 * acc), sprintf('%d / %d', sum(reg(pm(icf)) == 2), sum(icf))};
    if ~isempty(iR) && m ~= iR
        e  = mean(abs(R.oof{m}(:, test) - yt'), 1)';
        eR = mean(abs(R.oof{iR}(:, test) - yt'), 1)';
        [d, lo, hi] = local_boot(e - eR, grp, nboot);
        verdict = 'n.s.'; if hi < 0, verdict = 'better'; elseif lo > 0, verdict = 'worse'; end
        rows(m, 7:9) = {d, sprintf('[%.3f, %.3f]', lo, hi), verdict};
    else
        rows(m, 7:9) = {NaN, '', ''};
    end
end
R.table = rows; R.head = head;
R.yt = yt; R.test = test; R.reg = reg;
end

function s = local_sd(v)
if numel(v) > 1, s = std(v); else, s = NaN; end
end

function [d, lo, hi] = local_boot(diff, grp, B)
% Bootstrap over formulas (all entries of a composition resampled together).
% Vectorised: all B resamples are drawn in one block (same random stream as
% B sequential draws) and each resample mean is computed from per-formula
% sums and counts weighted by how often each formula was drawn.
G = max(grp); d = mean(diff);
gs = accumarray(grp, diff(:), [G 1]);    % sum of differences per formula
gn = accumarray(grp, 1, [G 1]);          % entries per formula
if exist('RandStream', 'class') == 8
    rs = RandStream('mt19937ar', 'Seed', 12345); pick = randi(rs, G, G, B);
else                                     % Octave
    rand('twister', 12345); pick = randi(G, G, B); %#ok<RAND>
end
cnt = sparse(pick(:), repelem((1:B)', G), 1, G, B);   % times formula g drawn in resample b
bs = sort(full((gs' * cnt) ./ (gn' * cnt)))';
lo = bs(max(1, round(0.025 * B))); hi = bs(min(B, round(0.975 * B)));
end
