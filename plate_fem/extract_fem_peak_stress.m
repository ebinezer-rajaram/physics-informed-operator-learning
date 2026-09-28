% extract_fem_peak_stress.m
% Run this after Plate_hole.m has executed (stress_full must be in workspace)
% OR re-run Plate_hole.m with this appended at the end.

% ---- FEM peak sigma_11 ----
sigma11_fem = stress_full(:, 1);          % column 1 of stress_full is sigma_11
[peak_fem, idx_peak] = max(sigma11_fem);

% centroid of peak element
asm_peak = connectivities(idx_peak, 1:3);
x1_peak  = mean(x1(asm_peak));
x2_peak  = mean(x2(asm_peak));

fprintf('FEM peak sigma_11 = %.4f N/m^2\n', peak_fem);
fprintf('  at element centroid (x1, x2) = (%.4f, %.4f)\n', x1_peak, x2_peak);

% ---- Kirsch analytical reference (infinite plate) ----
sigma_inf = 0.1;          % applied far-field stress
SCF_kirsch = 3.0;
sigma11_kirsch = SCF_kirsch * sigma_inf;
fprintf('Kirsch (infinite plate) peak sigma_11 = %.4f N/m^2\n', sigma11_kirsch);

% ---- PINN prediction ----
sigma11_pinn = 0.216;     % replace with your actual PINN peak
fprintf('PINN peak sigma_11 = %.4f N/m^2\n', sigma11_pinn);

% ---- Errors ----
err_pinn_vs_fem    = abs(sigma11_pinn   - peak_fem) / peak_fem * 100;
err_fem_vs_kirsch  = abs(peak_fem       - sigma11_kirsch) / sigma11_kirsch * 100;
err_pinn_vs_kirsch = abs(sigma11_pinn   - sigma11_kirsch) / sigma11_kirsch * 100;

fprintf('\n--- Comparison ---\n');
fprintf('FEM vs Kirsch:  %.1f%% below infinite-plate SCF=3 (finite domain effect)\n', err_fem_vs_kirsch);
fprintf('PINN vs FEM:    %.1f%% error relative to FEM reference\n', err_pinn_vs_fem);
fprintf('PINN vs Kirsch: %.1f%% error relative to Kirsch bound\n', err_pinn_vs_kirsch);

% ---- Optional: save for report ----
save('peak_stress_comparison.mat', 'peak_fem', 'x1_peak', 'x2_peak', ...
     'sigma11_kirsch', 'sigma11_pinn', ...
     'err_pinn_vs_fem', 'err_fem_vs_kirsch', 'err_pinn_vs_kirsch');
