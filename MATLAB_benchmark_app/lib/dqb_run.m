function R = dqb_run(D, cfg, progress)
%DQB_RUN  Cross-validated benchmark of several models on dataset D.
%   cfg.models   cellstr of model names
%   cfg.scheme   'compositions' | 'spacegroups' | 'sites' | 'kfold' | 'transfer'
%   cfg.folds    number of folds (default 10)
%   cfg.repeats  number of prespecified partitions (default 10)
%   cfg.seed     first random seed (partition r uses seed + r - 1)
%   cfg.featIdx  indices of the descriptors to use (default all)
%   cfg.engine   'MATLAB' (native models) or 'Python' (code of the paper)
%   cfg.parallel true (default) runs all model x partition x fold fits in
%                parallel (process pool / joblib); results are identical
%                to a serial run because every fit sets its own seed
%   progress     optional function handle progress(fraction, message)
%   R.oof{m}     repeats x n out-of-fold predictions of model m

if nargin < 3 || isempty(progress), progress = @(f, msg) []; end
cfg = local_defaults(cfg, D);
if any(strcmp(cfg.scheme, {'sites', 'transfer'})), cfg.repeats = 1; end

R = struct('y', D.y, 'models', {cfg.models}, 'cfg', cfg, 'D', D);
R.oof = cell(1, numel(cfg.models)); R.sd = R.oof;

if strcmpi(cfg.engine, 'Python')
    R = local_python(R, D, cfg, progress);
else
    X = D.X(:, cfg.featIdx); feat = D.feat(cfg.featIdx);
    splits = cell(1, cfg.repeats);
    for r = 1:cfg.repeats, splits{r} = dqb_splits(D, cfg.scheme, cfg.folds, cfg.seed + r - 1); end
    % one task per (model, partition, fold); slowest models submitted first
    [~, mo] = sort(cellfun(@dqb_cost, cfg.models), 'descend');
    task = zeros(0, 3);
    for m = mo(:)'
        for r = 1:cfg.repeats
            for k = 1:numel(splits{r}), task(end + 1, :) = [m r k]; end %#ok<AGROW>
        end
    end
    T = size(task, 1); args = cell(1, T); labels = cell(1, T);
    for t = 1:T
        m = task(t, 1); r = task(t, 2); k = task(t, 3); te = splits{r}{k};
        args{t} = {cfg.models{m}, X(~te, :), D.y(~te), X(te, :), feat, cfg.seed + r - 1};
        labels{t} = sprintf('%s  partition %d/%d  fold %d/%d', cfg.models{m}, r, cfg.repeats, k, numel(splits{r}));
    end
    out = dqb_parmap(@dqb_fit_task, args, cfg.parallel, progress, labels);
    for m = 1:numel(cfg.models)
        R.oof{m} = nan(cfg.repeats, D.n); R.sd{m} = R.oof{m};
    end
    for t = 1:T
        m = task(t, 1); r = task(t, 2); te = splits{r}{task(t, 3)};
        R.oof{m}(r, te) = out{t}.p; R.sd{m}(r, te) = out{t}.s;
    end
end
R = dqb_summary(R);
end

% ---------------------------------------------------------------------
function cfg = local_defaults(cfg, D)
def = struct('folds', 10, 'repeats', 10, 'seed', 0, 'engine', 'MATLAB', 'featIdx', 1:numel(D.feat), ...
    'python', 'python', 'codeDir', '', 'scheme', 'compositions', 'parallel', true);
f = fieldnames(def);
for i = 1:numel(f)
    if ~isfield(cfg, f{i}) || isempty(cfg.(f{i})), cfg.(f{i}) = def.(f{i}); end
end
end

function R = local_python(R, D, cfg, progress)
here = fileparts(mfilename('fullpath'));
out = tempname; mkdir(out);
job = struct('code_dir', cfg.codeDir, 'data', D.path, 'out_dir', out, 'scheme', cfg.scheme, ...
    'models', {cfg.models}, 'features', {D.feat(cfg.featIdx)}, 'folds', cfg.folds, ...
    'repeats', cfg.repeats, 'seed', cfg.seed, 'states', [], 'n_jobs', 1 - 2 * logical(cfg.parallel));   % -1 = all cores
jf = fullfile(out, 'job.json');
fid = fopen(jf, 'w'); fwrite(fid, jsonencode(job), 'char'); fclose(fid);
progress(0.02, 'Running the Python benchmark code ...');
cmd = sprintf('"%s" "%s" "%s"', cfg.python, fullfile(here, 'matlab_bridge.py'), jf);
if ispc, cmd = ['"' cmd '"']; end      % cmd.exe /c strips the outer pair of quotes
[st, msg] = system(cmd);
if st ~= 0, error('dqb:python', 'Python run failed:\n%s', msg); end
for m = 1:numel(cfg.models)
    R.oof{m} = dqb_readcsv(fullfile(out, ['oof_' cfg.models{m} '.csv']));
    R.sd{m}  = dqb_readcsv(fullfile(out, ['std_' cfg.models{m} '.csv']));
    if isvector(R.oof{m}), R.oof{m} = R.oof{m}(:)'; R.sd{m} = R.sd{m}(:)'; end
end
progress(1, 'Python run finished');
end
