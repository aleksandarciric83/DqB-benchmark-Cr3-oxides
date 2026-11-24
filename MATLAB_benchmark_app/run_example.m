%RUN_EXAMPLE  Command-line use of the benchmark functions (no GUI).
here = fileparts(mfilename('fullpath')); addpath(fullfile(here, 'lib'));
D = dqb_load(fullfile(here, '..', '..', 'Code', 'data', 'train183.xlsx'));   % data are kept only in Code/data

% 1) MATLAB-native models, unseen compositions, 10 x 10-fold
cfg = struct('models', {{'Ridge', 'Random Forest', 'SVR', 'Constant'}}, ...
             'scheme', 'compositions', 'folds', 10, 'repeats', 10, 'seed', 0);
R = dqb_run(D, cfg, @(f, msg) fprintf('%5.1f %%  %s\n', 100 * f, msg));
disp(cell2table(R.table, 'VariableNames', matlab.lang.makeValidName(R.head)));

% 2) Models of the paper through Python (needs Python with requirements.txt)
% cfg.engine = 'Python'; cfg.python = 'python'; cfg.codeDir = fullfile(here, '..', '..', 'Code');
% cfg.models = {'CatBoost', 'RandomForest', 'SVR', 'Ridge', 'Constant'};
% R = dqb_run(D, cfg);

% 3) Original evaluation procedure (selection of the most favourable of 31 partitions)
% P = dqb_partition(D, struct('models', {{'Ridge', 'Random Forest'}}));
% disp(cell2table(P.table, 'VariableNames', matlab.lang.makeValidName(P.head)));
