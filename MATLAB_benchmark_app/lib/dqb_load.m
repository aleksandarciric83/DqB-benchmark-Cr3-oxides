function D = dqb_load(path)
%DQB_LOAD  Read a Dq/B dataset (xlsx or csv) into a struct.
%   Required column: 'Dq/B'. Recognised optional columns: 'Formula',
%   'SGR No.' (space group), 'Cr_site' (substituted cation), 'Subset'
%   ('Kumar'/'Inherited' vs 'Added'). All other numeric columns are
%   offered as descriptors.

[~, ~, ext] = fileparts(path);
if exist('readtable', 'file') == 2 || exist('readtable', 'builtin') == 5
    if any(strcmpi(ext, {'.xlsx', '.xls'}))
        T = readtable(path, 'VariableNamingRule', 'preserve');
    else
        T = readtable(path, 'VariableNamingRule', 'preserve', 'Delimiter', ',');
    end
    names = T.Properties.VariableNames;
    getcol = @(nm) T.(nm);
else                                   % Octave fallback (csv only)
    [names, cols] = local_readcsv(path);
    getcol = @(nm) cols{strcmp(names, nm)};
end

D = struct();
D.path = path;
iy = find(strcmpi(names, 'Dq/B'), 1);
if isempty(iy), error('dqb:load', 'Column "Dq/B" not found.'); end
D.y = double(getcol(names{iy}));
n = numel(D.y);

D.formula = local_text(names, getcol, 'Formula', n);
D.site    = local_text(names, getcol, 'Cr_site', n);
D.subset  = local_text(names, getcol, 'Subset', n);
isg = find(strcmpi(names, 'SGR No.'), 1);
if isempty(isg), D.sg = nan(n, 1); else, D.sg = double(getcol(names{isg})); end

skip = {'Dq/B', 'Formula', 'Cr_site', 'Subset', 'No.', 'max_First ionization energy (kJ/mol)'};
feat = {}; X = [];
for k = 1:numel(names)
    if any(strcmpi(names{k}, skip)), continue; end
    v = getcol(names{k});
    if isnumeric(v) && all(isfinite(double(v)))
        feat{end + 1} = names{k}; %#ok<AGROW>
        X = [X, double(v)];       %#ok<AGROW>
    end
end
D.X = X;
D.feat = feat;
D.n = n;
end

% ---------------------------------------------------------------------
function c = local_text(names, getcol, nm, n)
i = find(strcmpi(names, nm), 1);
if isempty(i), c = repmat({''}, n, 1); return; end
v = getcol(names{i});
if isnumeric(v), c = cellstr(num2str(v(:)));
elseif isstring(v), c = cellstr(v(:));
else, c = v(:);
end
c = cellfun(@(s) strtrim(char(s)), c, 'UniformOutput', false);
end

function [names, cols] = local_readcsv(path)
fid = fopen(path, 'r'); hdr = fgetl(fid);
names = local_split(hdr);
rows = {};
while true
    l = fgetl(fid); if ~ischar(l), break; end
    if isempty(strtrim(l)), continue; end
    rows{end + 1} = local_split(l); %#ok<AGROW>
end
fclose(fid);
cols = cell(1, numel(names));
for k = 1:numel(names)
    v = cellfun(@(r) r{k}, rows, 'UniformOutput', false)';
    d = str2double(v);
    if all(~isnan(d) | cellfun(@isempty, v)), cols{k} = d; else, cols{k} = v; end
end
end

function p = local_split(l)
p = {}; cur = ''; q = false;
for ch = l
    if ch == '"', q = ~q;
    elseif ch == ',' && ~q, p{end + 1} = cur; cur = ''; %#ok<AGROW>
    else, cur(end + 1) = ch; %#ok<AGROW>
    end
end
p{end + 1} = cur;
end
