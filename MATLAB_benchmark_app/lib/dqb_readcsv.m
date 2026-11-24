function A = dqb_readcsv(f)
%DQB_READCSV  Numeric matrix from a csv file (readmatrix if available).
if exist('readmatrix', 'file') == 2 || exist('readmatrix', 'builtin') == 5
    A = readmatrix(f);
else
    A = dlmread(f, ',');
end
end
