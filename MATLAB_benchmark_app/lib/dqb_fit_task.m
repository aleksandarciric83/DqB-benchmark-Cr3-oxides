function o = dqb_fit_task(model, Xtr, ytr, Xte, feat, seed)
%DQB_FIT_TASK  One train/predict task for DQB_PARMAP (single output struct).
[o.p, o.s] = dqb_fit_predict(model, Xtr, ytr, Xte, feat, seed);
end
