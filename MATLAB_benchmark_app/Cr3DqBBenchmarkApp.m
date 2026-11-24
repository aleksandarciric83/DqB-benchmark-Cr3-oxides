classdef Cr3DqBBenchmarkApp < handle
%CR3DQBBENCHMARKAPP  GUI for rerunning the Dq/B benchmark of
%   "Data Curation and Generalisation in Machine-Learning Prediction of
%   Dq/B for Cr3+-Doped Oxide Phosphors" (Đurković, Ćirić, Ristić, Dramićanin).
%
%   Start with:   app = Cr3DqBBenchmarkApp;
%
%   Two engines:
%     MATLAB  - native models (Statistics and Machine Learning Toolbox):
%               boosted trees (LSBoost), Random Forest, SVR, GPR, ridge, constant
%     Python  - calls the benchmark code of the paper (common185.py) and gives
%               the published models: CatBoost, XGBoost, LightGBM, Random Forest,
%               SVR, GPR, ridge, constant
%   Requires MATLAB R2020b or newer. With "Parallel computing" ticked, all
%   model x partition x fold fits run on a process pool (MATLAB engine,
%   Parallel Computing Toolbox) or on all cores through joblib (Python
%   engine); results are identical to a serial run.

    properties (Access = public)
        UIFigure
    end
    properties (Access = private)
        D = []                 % loaded dataset
        R = []                 % last benchmark result
        P = []                 % last partition-selection result
        % controls
        DataLabel; EngineDrop; PyPathEdit; CodeDirEdit; CodeDirBtn
        SchemeDrop; FoldsSpin; RepSpin; SeedSpin; ParCheck
        ModelList; FeatList; RunBtn; ExportBtn
        % outputs
        Tabs; ResTable; ResNote; ParityAx; ParityDrop; BarAx; DataTable; LogArea
    end

    properties (Constant, Access = private)
        SCHEMES = {'Unseen compositions (formula-grouped k-fold)', 'compositions'; ...
                   'Unseen space groups (space-group-grouped k-fold)', 'spacegroups'; ...
                   'Unseen substitution sites (leave-one-cation-out)', 'sites'; ...
                   'Ungrouped k-fold (as Kumar et al.)', 'kfold'; ...
                   'Transfer: inherited entries -> added hosts', 'transfer'; ...
                   'Partition selection (original procedure, 31 partitions)', 'partition'}
        MODELS_M = {'Boosted trees', 'Random Forest', 'SVR', 'GPR', 'Ridge', 'Constant'}
        MODELS_P = {'CatBoost', 'XGBoost', 'LightGBM', 'RandomForest', 'SVR', 'GPR', 'Ridge', 'Constant'}
        COL = [0.20 0.45 0.70; 0.85 0.55 0.10; 0.75 0.20 0.20]   % WCF, ICF, SCF
    end

    methods
        function app = Cr3DqBBenchmarkApp
            here = fileparts(mfilename('fullpath'));
            addpath(fullfile(here, 'lib'));
            app.build();
            def = fullfile(here, 'data', 'train183.xlsx');
            if ~exist(def, 'file'), def = fullfile(here, '..', '..', 'Code', 'data', 'train183.xlsx'); end   % package layout: data only in Code/data
            if exist(def, 'file'), app.loadData(def); end
        end
        function delete(app)
            if ~isempty(app.UIFigure) && isvalid(app.UIFigure), delete(app.UIFigure); end
        end
    end

    methods (Access = private)
        % ------------------------------------------------------------ layout
        function build(app)
            app.UIFigure = uifigure('Name', 'Cr3+ Dq/B benchmark', 'Position', [80 60 1320 800], ...
                'CloseRequestFcn', @(~, ~) delete(app));
            g = uigridlayout(app.UIFigure, [1 2]); g.ColumnWidth = {340, '1x'};

            L = uigridlayout(g, [22 2]); L.RowHeight = repmat({22}, 1, 22);
            L.RowHeight{16} = 110; L.RowHeight{18} = 150; L.ColumnWidth = {110, '1x'};
            L.RowSpacing = 5; L.Padding = [8 8 8 8];

            app.head(L, 1, 'Data');
            b = uibutton(L, 'Text', 'Load dataset (xlsx/csv) ...', 'ButtonPushedFcn', @(~, ~) app.onLoad());
            b.Layout.Row = 2; b.Layout.Column = [1 2];
            app.DataLabel = uilabel(L, 'Text', 'no dataset loaded', 'FontColor', [0.4 0.4 0.4]);
            app.DataLabel.Layout.Row = 3; app.DataLabel.Layout.Column = [1 2];

            app.head(L, 4, 'Engine');
            app.EngineDrop = uidropdown(L, 'Items', {'MATLAB (native models)', 'Python (models of the paper)'}, ...
                'ValueChangedFcn', @(~, ~) app.onEngine());
            app.EngineDrop.Layout.Row = 5; app.EngineDrop.Layout.Column = [1 2];
            lb = uilabel(L, 'Text', 'Python executable'); lb.Layout.Row = 6;
            app.PyPathEdit = uieditfield(L, 'Value', 'python'); app.PyPathEdit.Layout.Row = 6; app.PyPathEdit.Layout.Column = 2;
            app.CodeDirBtn = uibutton(L, 'Text', 'Benchmark code folder ...', 'ButtonPushedFcn', @(~, ~) app.onCodeDir());
            app.CodeDirBtn.Layout.Row = 7; app.CodeDirBtn.Layout.Column = 1;
            app.CodeDirEdit = uieditfield(L, 'Value', ''); app.CodeDirEdit.Layout.Row = 7; app.CodeDirEdit.Layout.Column = 2;
            here = fileparts(mfilename('fullpath'));
            if exist(fullfile(here, 'python', 'common185.py'), 'file'), app.CodeDirEdit.Value = fullfile(here, 'python');
            elseif exist(fullfile(here, '..', '..', 'Code', 'common185.py'), 'file'), app.CodeDirEdit.Value = fullfile(here, '..', '..', 'Code'); end

            app.head(L, 8, 'Validation');
            app.SchemeDrop = uidropdown(L, 'Items', app.SCHEMES(:, 1)', 'ValueChangedFcn', @(~, ~) app.onScheme());
            app.SchemeDrop.Layout.Row = 9; app.SchemeDrop.Layout.Column = [1 2];
            lb = uilabel(L, 'Text', 'Folds'); lb.Layout.Row = 10;
            app.FoldsSpin = uispinner(L, 'Value', 10, 'Limits', [2 50], 'Step', 1); app.FoldsSpin.Layout.Row = 10; app.FoldsSpin.Layout.Column = 2;
            lb = uilabel(L, 'Text', 'Partitions (repeats)'); lb.Layout.Row = 11;
            app.RepSpin = uispinner(L, 'Value', 10, 'Limits', [1 100], 'Step', 1); app.RepSpin.Layout.Row = 11; app.RepSpin.Layout.Column = 2;
            lb = uilabel(L, 'Text', 'First seed'); lb.Layout.Row = 12;
            app.SeedSpin = uispinner(L, 'Value', 0, 'Limits', [0 1e6], 'Step', 1); app.SeedSpin.Layout.Row = 12; app.SeedSpin.Layout.Column = 2;
            hasPCT = ~isempty(ver('parallel')) && license('test', 'Distrib_Computing_Toolbox');
            app.ParCheck = uicheckbox(L, 'Text', sprintf('Parallel computing (%d cores)', feature('numcores')), ...
                'Value', true, 'Tooltip', ['MATLAB engine: process pool (Parallel Computing Toolbox' ...
                local_if(hasPCT, '', ', not installed - runs serially') '). Python engine: joblib on all cores.']);
            app.ParCheck.Layout.Row = 13; app.ParCheck.Layout.Column = [1 2];

            app.head(L, 15, 'Models (Ctrl/Shift for several)');
            app.ModelList = uilistbox(L, 'Items', app.MODELS_M, 'Multiselect', 'on', 'Value', app.MODELS_M);
            app.ModelList.Layout.Row = 16; app.ModelList.Layout.Column = [1 2];
            app.head(L, 17, 'Descriptors');
            app.FeatList = uilistbox(L, 'Items', {}, 'Multiselect', 'on');
            app.FeatList.Layout.Row = 18; app.FeatList.Layout.Column = [1 2];

            app.RunBtn = uibutton(L, 'Text', 'Run benchmark', 'FontWeight', 'bold', ...
                'BackgroundColor', [0.18 0.45 0.30], 'FontColor', [1 1 1], 'ButtonPushedFcn', @(~, ~) app.onRun());
            app.RunBtn.Layout.Row = [20 21]; app.RunBtn.Layout.Column = [1 2];
            app.ExportBtn = uibutton(L, 'Text', 'Export results to xlsx ...', 'Enable', 'off', 'ButtonPushedFcn', @(~, ~) app.onExport());
            app.ExportBtn.Layout.Row = 22; app.ExportBtn.Layout.Column = [1 2];

            % ---- right: tabs
            app.Tabs = uitabgroup(g);
            t1 = uitab(app.Tabs, 'Title', 'Results');
            g1 = uigridlayout(t1, [2 1]); g1.RowHeight = {'1x', 70};
            app.ResTable = uitable(g1, 'ColumnName', {}, 'RowName', {});
            app.ResNote = uitextarea(g1, 'Editable', 'off', 'Value', {''}, 'FontSize', 11);

            t2 = uitab(app.Tabs, 'Title', 'Parity plot');
            g2 = uigridlayout(t2, [2 3]); g2.RowHeight = {24, '1x'}; g2.ColumnWidth = {45, 260, '1x'};
            g2.Padding = [10 6 10 6];
            lb = uilabel(g2, 'Text', 'Model'); lb.Layout.Row = 1; lb.Layout.Column = 1;
            app.ParityDrop = uidropdown(g2, 'Items', {'-'}, 'ValueChangedFcn', @(~, ~) app.plotParity());
            app.ParityDrop.Layout.Row = 1; app.ParityDrop.Layout.Column = 2;
            app.ParityAx = uiaxes(g2); app.ParityAx.Layout.Row = 2; app.ParityAx.Layout.Column = [1 3];   % full tab width

            t3 = uitab(app.Tabs, 'Title', 'Model comparison');
            g3 = uigridlayout(t3, [1 1]); app.BarAx = uiaxes(g3);

            t4 = uitab(app.Tabs, 'Title', 'Dataset');
            g4 = uigridlayout(t4, [1 1]); app.DataTable = uitable(g4);

            t5 = uitab(app.Tabs, 'Title', 'Log');
            g5 = uigridlayout(t5, [1 1]); app.LogArea = uitextarea(g5, 'Editable', 'off', 'Value', {''});

            app.onEngine();
        end

        function head(~, L, row, txt)
            h = uilabel(L, 'Text', txt, 'FontWeight', 'bold');
            h.Layout.Row = row; h.Layout.Column = [1 2];
        end

        % ------------------------------------------------------------ callbacks
        function onLoad(app)
            [f, p] = uigetfile({'*.xlsx;*.xls;*.csv', 'Dq/B dataset'}, 'Select dataset');
            figure(app.UIFigure);
            if isequal(f, 0), return; end
            app.loadData(fullfile(p, f));
        end

        function loadData(app, path)
            try
                app.D = dqb_load(path);
            catch e
                uialert(app.UIFigure, e.message, 'Could not load dataset'); return;
            end
            D = app.D; [~, f, x] = fileparts(path);
            nK = sum(ismember(lower(D.subset), {'kumar', 'inherited'}));
            app.DataLabel.Text = sprintf('%s%s: %d entries, %d descriptors%s', f, x, D.n, numel(D.feat), ...
                local_if(nK > 0, sprintf(' (%d inherited, %d added)', nK, D.n - nK), ''));
            app.FeatList.Items = D.feat; app.FeatList.Value = D.feat;
            T = array2table([D.y D.X], 'VariableNames', [{'Dq/B'}, D.feat]);
            T = [table(D.formula, D.subset, D.site, 'VariableNames', {'Formula', 'Subset', 'Cr site'}) T];
            app.DataTable.Data = T;
            app.log(sprintf('Loaded %s (%d entries).', path, D.n));
        end

        function onEngine(app)
            py = startsWith(app.EngineDrop.Value, 'Python');
            app.ModelList.Items = local_if(py, app.MODELS_P, app.MODELS_M);
            app.ModelList.Value = app.ModelList.Items;
            en = local_if(py, 'on', 'off');
            app.PyPathEdit.Enable = en; app.CodeDirEdit.Enable = en; app.CodeDirBtn.Enable = en;
        end

        function onCodeDir(app)
            d = uigetdir(app.CodeDirEdit.Value, 'Folder containing common185.py');
            figure(app.UIFigure);
            if ~isequal(d, 0), app.CodeDirEdit.Value = d; end
        end

        function onScheme(app)
            sc = app.scheme();
            app.FoldsSpin.Enable = local_if(any(strcmp(sc, {'sites', 'transfer', 'partition'})), 'off', 'on');
            app.RepSpin.Enable = local_if(any(strcmp(sc, {'sites', 'transfer', 'partition'})), 'off', 'on');
            if strcmp(sc, 'spacegroups'), app.RepSpin.Value = 5; elseif strcmp(sc, 'compositions'), app.RepSpin.Value = 10; end
        end

        function sc = scheme(app)
            sc = app.SCHEMES{strcmp(app.SCHEMES(:, 1), app.SchemeDrop.Value), 2};
        end

        function onRun(app)
            if isempty(app.D), uialert(app.UIFigure, 'Load a dataset first.', 'No data'); return; end
            if isempty(app.ModelList.Value), uialert(app.UIFigure, 'Select at least one model.', 'No model'); return; end
            if isempty(app.FeatList.Value), uialert(app.UIFigure, 'Select at least one descriptor.', 'No descriptor'); return; end
            cfg = struct();
            cfg.models = app.ModelList.Value; cfg.scheme = app.scheme();
            cfg.folds = app.FoldsSpin.Value; cfg.repeats = app.RepSpin.Value; cfg.seed = app.SeedSpin.Value;
            cfg.featIdx = find(ismember(app.D.feat, app.FeatList.Value));
            cfg.engine = local_if(startsWith(app.EngineDrop.Value, 'Python'), 'Python', 'MATLAB');
            cfg.python = app.PyPathEdit.Value; cfg.codeDir = app.CodeDirEdit.Value;
            cfg.parallel = app.ParCheck.Value;
            if strcmp(cfg.engine, 'Python') && ~exist(fullfile(cfg.codeDir, 'common185.py'), 'file')
                uialert(app.UIFigure, 'Select the folder that contains common185.py (the python/ folder of the app or the code folder of the package).', 'Benchmark code not found'); return;
            end
            dlg = uiprogressdlg(app.UIFigure, 'Title', 'Running benchmark', 'Message', 'Starting ...', 'Cancelable', 'on');
            prog = @(f, msg) app.progress(dlg, f, msg);
            app.RunBtn.Enable = 'off'; t0 = tic;
            app.log(sprintf('Run: %s%s | %s | %s | models: %s | %d descriptors', cfg.engine, ...
                local_if(cfg.parallel, ' (parallel)', ' (serial)'), app.SchemeDrop.Value, ...
                local_if(any(strcmp(cfg.scheme, {'compositions', 'spacegroups', 'kfold'})), sprintf('%d-fold x %d', cfg.folds, cfg.repeats), ''), ...
                strjoin(cfg.models, ', '), numel(cfg.featIdx)));
            try
                if strcmp(cfg.scheme, 'partition')
                    app.P = dqb_partition(app.D, cfg, prog); app.R = [];
                    app.showPartition();
                else
                    app.R = dqb_run(app.D, cfg, prog); app.P = [];
                    app.showResults();
                end
                app.ExportBtn.Enable = 'on';
                app.log(sprintf('Finished in %.0f s.', toc(t0)));
            catch e
                if strcmp(e.identifier, 'dqb:cancel'), app.log('Run cancelled.');
                else, uialert(app.UIFigure, e.message, 'Run failed'); app.log(['Error: ' e.message]); end
            end
            close(dlg); app.RunBtn.Enable = 'on';
        end

        function progress(~, dlg, f, msg)
            if dlg.CancelRequested, error('dqb:cancel', 'cancelled'); end
            dlg.Value = min(max(f, 0), 1); dlg.Message = msg; drawnow limitrate;
        end

        % ------------------------------------------------------------ display
        function showResults(app)
            R = app.R; C = R.table;
            fmt = {'%s', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%+.3f', '%s', '%s', '%s', '%s'};
            out = C;
            for i = 1:size(C, 1)
                for j = 2:size(C, 2)
                    v = C{i, j};
                    if isnumeric(v), if isnan(v), out{i, j} = ''; else, out{i, j} = sprintf(fmt{j}, v); end, end
                end
            end
            app.ResTable.Data = out; app.ResTable.ColumnName = R.head;
            nt = sum(R.test);
            app.ResNote.Value = {sprintf(['Pooled out-of-fold metrics on %d entries, mean and SD over %d partition(s). ' ...
                'dMAE vs ridge: per-entry absolute error averaged over partitions, difference to ridge regression, ' ...
                '95%% interval from 4000 bootstrap resamples over formulas (better/worse: interval excludes zero). ' ...
                'Regimes: WCF < 2.23 <= ICF <= 2.42 < SCF; classification from the partition-averaged prediction.'], nt, size(R.oof{1}, 1))};
            app.ParityDrop.Items = R.models; app.ParityDrop.Value = R.models{1};
            app.plotParity(); app.plotBars();
            app.Tabs.SelectedTab = app.Tabs.Children(1);
        end

        function showPartition(app)
            P = app.P; C = P.table; out = C;
            for i = 1:size(C, 1)
                for j = 2:size(C, 2)
                    if j == size(C, 2), out{i, j} = sprintf('%d', C{i, j});
                    elseif any(j == [9 10]), out{i, j} = sprintf('%+.3f', C{i, j});
                    else, out{i, j} = sprintf('%.3f', C{i, j}); end
                end
            end
            app.ResTable.Data = out; app.ResTable.ColumnName = P.head;
            app.ResNote.Value = {['Original procedure: ungrouped 10-fold cross-validation for 31 random partitions ' ...
                '(states 5:5:100 and 5:7:100); the partition with the highest mean per-fold R2 is selected. ' ...
                'Averaged: mean and SD over all 31 partitions. Gain = selected - averaged.']};
            cla(app.BarAx); hold(app.BarAx, 'on');
            for m = 1:numel(P.models)
                A = P.raw{m};
                scatter(app.BarAx, m + 0.12 * randn(size(A, 1), 1), A(:, 3), 18, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.5);
                [~, b] = max(A(:, 2));
                plot(app.BarAx, m, A(b, 3), 'p', 'MarkerSize', 14, 'MarkerFaceColor', [0.85 0.65 0.1], 'MarkerEdgeColor', 'k');
                errorbar(app.BarAx, m + 0.3, mean(A(:, 3)), std(A(:, 3)), 'ko', 'MarkerFaceColor', 'k');
            end
            hold(app.BarAx, 'off'); xticks(app.BarAx, 1:numel(P.models)); xticklabels(app.BarAx, P.models);
            ylabel(app.BarAx, 'Pooled R^2'); title(app.BarAx, 'Pooled R^2 of the 31 partitions (star: selected, dot: mean \pm SD)');
            grid(app.BarAx, 'on'); xlim(app.BarAx, [0.5 numel(P.models) + 0.5]);
            app.ParityDrop.Items = {'-'}; cla(app.ParityAx);
        end

        function plotParity(app)
            if isempty(app.R), return; end
            R = app.R; m = find(strcmp(R.models, app.ParityDrop.Value), 1); if isempty(m), return; end
            ax = app.ParityAx; cla(ax, 'reset'); hold(ax, 'on');
            y = R.yt; O = R.oof{m}(:, R.test); pm = mean(O, 1)'; ps = std(O, 0, 1)';
            v = [y; pm; pm - ps; pm + ps]; v = v(isfinite(v));
            pad = 0.05 * (max(v) - min(v)); lim = [min(v) - pad, max(v) + pad];
            hW = fill(ax, [lim(1) lim(2) lim(2) lim(1)], [2.23 2.23 2.42 2.42], [1 0.93 0.8], ...
                'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', 'ICF window (2.23-2.42)');
            for b = [2.23 2.42]
                xline(ax, b, ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
            end
            hL = plot(ax, lim, lim, 'k-', 'LineWidth', 1, 'DisplayName', 'y = x');
            rg = R.reg(y); names = {'WCF', 'ICF', 'SCF'}; hS = gobjects(0);
            for k = 1:3
                i = rg == k; if ~any(i), continue; end
                if size(O, 1) > 1
                    errorbar(ax, y(i), pm(i), ps(i), 'LineStyle', 'none', 'CapSize', 0, ...
                        'Color', 0.45 + 0.55 * app.COL(k, :), 'HandleVisibility', 'off');
                end
                hS(end + 1) = scatter(ax, y(i), pm(i), 46, app.COL(k, :), 'filled', 'MarkerEdgeColor', 'w', ...
                    'LineWidth', 0.5, 'DisplayName', sprintf('%s (n = %d)', names{k}, sum(i))); %#ok<AGROW>
            end
            hold(ax, 'off'); axis(ax, 'square'); xlim(ax, lim); ylim(ax, lim);
            box(ax, 'on'); grid(ax, 'on'); ax.GridAlpha = 0.12; ax.FontSize = 12; ax.LineWidth = 0.8;
            xlabel(ax, 'Experimental Dq/B'); ylabel(ax, 'Predicted Dq/B (out-of-fold)');
            q = dqb_metrics(y, pm); c = polyfit(y, pm, 1);
            title(ax, R.models{m}, 'FontSize', 14);
            subtitle(ax, sprintf('R^2 = %.3f    MAE = %.3f    slope = %.2f    regime accuracy %s%s', q.R2, q.MAE, c(1), ...
                R.table{m, 10}, local_if(size(O, 1) > 1, sprintf('    (bars: SD over %d partitions)', size(O, 1)), '')), ...
                'FontSize', 11, 'Color', [0.3 0.3 0.3]);
            legend(ax, [hW hL hS], 'Location', 'northwest', 'FontSize', 11, 'Box', 'off');
        end

        function plotBars(app)
            R = app.R; ax = app.BarAx; cla(ax); M = numel(R.models);
            mae = cellfun(@(v) mean(v(:, 2)), R.metrics); sd = cellfun(@(v) local_sd(v(:, 2)), R.metrics);
            r2 = cellfun(@(v) mean(v(:, 1)), R.metrics);
            b = bar(ax, mae, 'FaceColor', [0.30 0.55 0.45]); hold(ax, 'on');
            if any(~isnan(sd)), errorbar(ax, 1:M, mae, sd, 'k.', 'LineWidth', 1); end
            for m = 1:M, text(ax, m, mae(m) + 0.006, sprintf('R^2 %.2f', r2(m)), 'HorizontalAlignment', 'center', 'FontSize', 9); end
            iC = find(strcmp(R.models, 'Constant'), 1);
            if ~isempty(iC), yline(ax, mae(iC), '--', 'constant predictor'); end
            hold(ax, 'off'); xticks(ax, 1:M); xticklabels(ax, R.models); ylabel(ax, 'MAE (Dq/B)');
            ylim(ax, [0 max(mae + local_nz(sd)) * 1.15]); grid(ax, 'on'); b.BarWidth = 0.6;
            title(ax, sprintf('%s: MAE (mean \\pm SD over partitions)', app.SchemeDrop.Value));
        end

        % ------------------------------------------------------------ export / log
        function onExport(app)
            [f, p] = uiputfile('*.xlsx', 'Save results', 'Dq_B_benchmark_results.xlsx');
            figure(app.UIFigure);
            if isequal(f, 0), return; end
            fn = fullfile(p, f);
            if exist(fn, 'file'), delete(fn); end
            info = {'Dataset', app.D.path; 'Engine', app.EngineDrop.Value; 'Validation', app.SchemeDrop.Value; ...
                    'Folds', app.FoldsSpin.Value; 'Partitions', app.RepSpin.Value; 'First seed', app.SeedSpin.Value; ...
                    'Parallel', local_if(app.ParCheck.Value, 'yes', 'no'); ...
                    'Descriptors', strjoin(app.FeatList.Value, '; '); 'Date', char(datetime('now'))};
            writecell(info, fn, 'Sheet', 'Settings');
            if ~isempty(app.R)
                writecell([app.R.head; app.R.table], fn, 'Sheet', 'Summary');
                pr = [{'Formula', 'Dq/B exp'}, strcat(app.R.models, ' (mean OOF)')];
                M = [app.R.yt, cell2mat(cellfun(@(o) mean(o(:, app.R.test), 1)', app.R.oof, 'UniformOutput', false))];
                writecell([pr; [app.D.formula(app.R.test), num2cell(M)]], fn, 'Sheet', 'Predictions');
                for m = 1:numel(app.R.models)
                    v = app.R.metrics{m};
                    writecell([{'Partition', 'R2', 'MAE', 'RMSE'}; num2cell([(1:size(v, 1))' v])], fn, 'Sheet', ['Partitions_' app.R.models{m}]);
                end
            else
                writecell([app.P.head; app.P.table], fn, 'Sheet', 'Partition selection');
                for m = 1:numel(app.P.models)
                    writecell([{'State', 'Per-fold R2', 'Pooled R2', 'MAE'}; num2cell(app.P.raw{m})], fn, 'Sheet', ['States_' app.P.models{m}]);
                end
            end
            app.log(['Results written to ' fn]);
        end

        function log(app, msg)
            app.LogArea.Value = [app.LogArea.Value(:); {sprintf('[%s] %s', char(datetime('now', 'Format', 'HH:mm:ss')), msg)}];
        end
    end
end

% ---------------------------------------------------------------- helpers
function v = local_if(c, a, b)
if c, v = a; else, v = b; end
end
function s = local_sd(v)
if numel(v) > 1, s = std(v); else, s = NaN; end
end
function v = local_nz(v)
v(isnan(v)) = 0;
end
