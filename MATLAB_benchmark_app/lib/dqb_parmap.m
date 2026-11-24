function out = dqb_parmap(fn, args, usePar, progress, labels)
%DQB_PARMAP  Evaluate out{t} = fn(args{t}{:}) for t = 1..T, in parallel
%   (parfeval on a process pool) when usePar is true and a pool is
%   available, otherwise serially. Tasks are submitted in the given order,
%   so put the slowest ones first for good load balancing.
%   progress(fraction, message) is called as tasks finish and is polled
%   every 0.25 s while waiting, so it may throw to cancel the run; any
%   queued or running tasks are then cancelled.

T = numel(args); out = cell(1, T);
if nargin < 4 || isempty(progress), progress = @(f, msg) []; end
if nargin < 5 || isempty(labels), labels = repmat({''}, 1, T); end
if T == 0, return; end

pool = [];
if usePar, pool = dqb_pool(progress); end

if isempty(pool)                                   % serial
    for t = 1:T
        out{t} = fn(args{t}{:});
        progress(t / T, labels{t});
    end
    return
end

F(1:T) = parallel.FevalFuture;
for t = 1:T, F(t) = parfeval(pool, fn, 1, args{t}{:}); end
cleanup = onCleanup(@() cancel(F));                %#ok<NASGU>  stops workers on error or cancel
nw = pool.NumWorkers; done = 0;
msg = sprintf('%d fits queued on %d workers', T, nw);
progress(0, msg);
while done < T
    [t, v] = fetchNext(F, 0.25);
    if isempty(t)
        progress(done / T, msg);                   % keeps the Cancel button responsive
        continue
    end
    out{t} = v; done = done + 1;
    msg = sprintf('%d / %d fits finished on %d workers   (last: %s)', done, T, nw, labels{t});
    progress(done / T, msg);
end
end
