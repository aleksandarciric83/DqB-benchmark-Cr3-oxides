function c = dqb_cost(model)
%DQB_COST  Rough relative cost of one fit, used only to submit the slowest
%   tasks first so that the parallel workers finish at about the same time.
switch model
    case {'SVR'},                                   c = 100;   % 90-point grid x 5 inner folds
    case {'GPR', 'GPR_raw1/r2'},                    c = 30;
    case {'CatBoost', 'CatBoost_catSG'},            c = 10;
    case {'Random Forest', 'RandomForest'},         c = 8;
    case {'Boosted trees', 'XGBoost', 'LightGBM'},  c = 4;
    otherwise,                                      c = 1;     % Ridge, Constant
end
end
