function pool = dqb_pool(progress)
%DQB_POOL  Return a process-based parallel pool, starting one if needed.
%   Returns [] when the Parallel Computing Toolbox is not available or the
%   pool cannot be started; callers then fall back to serial execution.
%   A thread pool is replaced by a process pool because TreeBagger,
%   fitrsvm and fitrgp are not all supported on thread workers.

if nargin < 1 || isempty(progress), progress = @(f, msg) []; end
pool = [];
if isempty(ver('parallel')) || ~license('test', 'Distrib_Computing_Toolbox'), return; end
try
    pool = gcp('nocreate');
    if ~isempty(pool) && isa(pool, 'parallel.ThreadPool')
        delete(pool); pool = [];
    end
    if isempty(pool)
        progress(0, 'Starting parallel pool (first run only, ~20 s) ...');
        try
            pool = parpool('Processes');
        catch
            pool = parpool('local');            % releases before R2022b
        end
        lib = fileparts(mfilename('fullpath'));
        wait(parfevalOnAll(pool, @addpath, 0, lib));
    end
catch e
    if strcmp(e.identifier, 'dqb:cancel'), rethrow(e); end
    warning('dqb:pool', 'Parallel pool unavailable (%s); running serially.', e.message);
    pool = [];
end
end
