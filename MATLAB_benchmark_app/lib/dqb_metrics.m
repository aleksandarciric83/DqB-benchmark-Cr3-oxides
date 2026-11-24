function m = dqb_metrics(y, p)
%DQB_METRICS  Pooled R^2, MAE and RMSE of predictions p for targets y.
y = y(:); p = p(:); ok = ~isnan(p);
y = y(ok); p = p(ok);
r = y - p;
m.R2 = 1 - sum(r.^2) / sum((y - mean(y)).^2);
m.MAE = mean(abs(r));
m.RMSE = sqrt(mean(r.^2));
end
