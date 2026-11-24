function P = dqb_partition(D, cfg, progress)
%DQB_PARTITION  Reproduce the original evaluation procedure of Kumar et al.:
%   ungrouped 10-fold cross-validation for 31 random partitions; the
%   partition with the highest mean per-fold R^2 is "selected". Compares
%   the selected partition with the average across all partitions.

if nargin < 3 || isempty(progress), progress = @(f, msg) []; end
states = unique([5:5:100, 5:7:100]);              % as in the original code (31 states)
if ~isfield(cfg, 'featIdx') || isempty(cfg.featIdx), cfg.featIdx = 1:numel(D.feat); end
if ~isfield(cfg, 'engine'), cfg.engine = 'MATLAB'; end
if ~isfield(cfg, 'parallel') || isempty(cfg.parallel), cfg.parallel = true; end
M = numel(cfg.models); X = D.X(:, cfg.featIdx); feat = D.feat(cfg.featIdx);
P.states = states; P.models = cfg.models; P.raw = cell(1, M);

if strcmpi(cfg.engine, 'Python')
    here = fileparts(mfilename('fullpath')); out = tempname; mkdir(out);
    job = struct('code_dir', cfg.codeDir, 'data', D.path, 'out_dir', out, 'scheme', 'partition', ...
        'models', {cfg.models}, 'features', {feat}, 'folds', 10, 'repeats', 1, 'seed', 0, 'states', states, ...
        'n_jobs', 1 - 2 * logical(cfg.parallel));   % -1 = all cores
    jf = fullfile(out, 'job.json'); fid = fopen(jf, 'w'); fwrite(fid, jsonencode(job), 'char'); fclose(fid);
    progress(0.02, 'Running the Python benchmark code ...');
    cmd = sprintf('"%s" "%s" "%s"', cfg.python, fullfile(here, 'matlab_bridge.py'), jf);
    if ispc, cmd = ['"' cmd '"']; end  % cmd.exe /c strips the outer pair of quotes
    [st, msg] = system(cmd);
    if st ~= 0, error('dqb:python', 'Python run failed:\n%s', msg); end
    for m = 1:M, P.raw{m} = dqb_readcsv(fullfile(out, ['partition_' cfg.models{m} '.csv'])); end
else
    S = numel(states); te = cell(1, S);
    for i = 1:S, te{i} = dqb_splits(D, 'kfold', 10, states(i)); end
    % one task per (model, state, fold); slowest models submitted first
    [~, mo] = sort(cellfun(@dqb_cost, cfg.models), 'descend');
    task = zeros(0, 3);
    for m = mo(:)'
        for i = 1:S
            for k = 1:numel(te{i}), task(end + 1, :) = [m i k]; end %#ok<AGROW>
        end
    end
    T = size(task, 1); args = cell(1, T); labels = cell(1, T);
    for t = 1:T
        m = task(t, 1); i = task(t, 2); k = task(t, 3); tk = te{i}{k};
        args{t} = {cfg.models{m}, X(~tk, :), D.y(~tk), X(tk, :), feat, states(i)};
        labels{t} = sprintf('%s  partition %d/%d  fold %d/10', cfg.models{m}, i, S, k);
    end
    out = dqb_parmap(@dqb_fit_task, args, cfg.parallel, progress, labels);
    oof = repmat({nan(D.n, 1)}, M, S); r2f = repmat({zeros(1, 10)}, M, S);
    for t = 1:T
        m = task(t, 1); i = task(t, 2); k = task(t, 3); tk = te{i}{k};
        oof{m, i}(tk) = out{t}.p; q = dqb_metrics(D.y(tk), out{t}.p); r2f{m, i}(k) = q.R2;
    end
    for m = 1:M
        A = zeros(S, 4);
        for i = 1:S
            q = dqb_metrics(D.y, oof{m, i});
            A(i, :) = [states(i), mean(r2f{m, i}(1:numel(te{i}))), q.R2, q.MAE];
        end
        P.raw{m} = A;
    end
end

P.head = {'Model', 'Selected: pooled R2', 'Selected: per-fold R2', 'Averaged: pooled R2', 'SD', ...
          'Averaged: per-fold R2', 'Min pooled', 'Max pooled', 'Gain pooled', 'Gain per-fold', 'Selected state'};
P.table = cell(M, numel(P.head));
for m = 1:M
    A = P.raw{m}; [~, b] = max(A(:, 2));
    P.table(m, :) = {cfg.models{m}, A(b, 3), A(b, 2), mean(A(:, 3)), std(A(:, 3)), mean(A(:, 2)), ...
        min(A(:, 3)), max(A(:, 3)), A(b, 3) - mean(A(:, 3)), A(b, 2) - mean(A(:, 2)), A(b, 1)};
end
end
