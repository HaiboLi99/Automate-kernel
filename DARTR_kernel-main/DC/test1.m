clear; close all; restoredefaultpath;  addpath('../'); add_mypaths;
rng(10);  plotOn = 1; saveON = 0; 


%%% 0 setttings: R_phi[u](x)  = \sum_r phi(r) * fun_g[u](x,r) dr, x in [0,1], r= [0,R0]
% the first example----Itegral operator
dx       = 0.005;                          % space mesh size in observation. 
N        = 10;                               %  number of data pairs (u_i,f_i)
u_Type   = 'stocFourier';        % types: 'Bspline', 'Fourier'; 'stocFourier';  'stocCosine' %  Fourier   
jump_disc = 0;               % jump discontinuity to increase rank of G 

R0       = 1;                % maximal interaction range [0,R0] for radial kernel
supp_u   = [-R0 1+R0];       % data u support    >>> f(x) with x = [0,1] 
example_type = 'LinearIntOpt';   % {'LinearIntOpt','nonlocal','Aggregation_StrForm'};
kernel_type  = 'sinx_smooth';       % Gaussian, sinkx, FracLap, sinx_smooth

  
% the second example-----Nonlocal operator
% dx       = 0.005;                          % space mesh size in observation. 
% N        = 10;                               %  number of data pairs (u_i,f_i)
% u_Type   = 'stocFourier';        % types: 'Bspline', 'Fourier', 'stocFourier';  'stocCosine' %  Fourier   
% jump_disc = 1;               % jump discontinuity to increase rank of G 
% 
% R0       = 1;                % maximal interaction range [0,R0] for radial kernel
% supp_u   = [-R0 1+R0];       % data u support    >>> f(x) with x = [0,1] 
% example_type = 'nonlocal';   % {'LinearIntOpt','nonlocal','Aggregation_StrForm'};
% kernel_type  = 'sinkx';       % Gaussian, sinkx, FracLap


% the third example-----Aggregation operator
% dx       = 0.005;                          % space mesh size in observation. 
% N        = 10;                             %  number of data pairs (u_i,f_i)
% u_Type   = 'randomDensity';        % types: 'Bspline', 'Fourier', 'stocFourier';  'stocCosine', randomDensity  
% jump_disc = 0;                     % jump discontinuity to increase rank of G 
% 
% R0       = 1;                    % maximal interaction range [0,R0] for radial kernel
% supp_u   = [-R0 1+R0];           % data u support    >>> f(x) with x = [0,1] 
% example_type = 'Aggregation_StrForm';   % {'LinearIntOpt','nonlocal','Aggregation_StrForm'};
% kernel_type  = 'sinx_cubic';       % Gaussian, sinkx, FracLap, sinx_smooth, polyx, powerFn, sinx_cubic



[kernelInfo, obsInfo]  = load_settings_v2(N, u_Type,jump_disc, supp_u, R0, dx,kernel_type,example_type);
noise_ratio            = 1;  obsInfo.noise_ratio  = noise_ratio;    
obsInfo.plotON = 1; % no plots in the parallel simulations. 

%% % 1. Generate data on x_mesh (adaptive to u, f) and pre-process data
integrator = 'quadgk'; %  'Riemann', 'quadgk'     % Riemann sum for checking. Quadgk for testing. 

[obsInfo,ux_val,fx_val]   = generateData2(kernelInfo, obsInfo, SAVE_DIR,saveON,integrator);    % get observation data
if strcmp(example_type, 'Aggregation_StrForm')
    noise_std_upperBd = max(round(obsInfo.noise_std_upperBd,2),0.1);
else
    noise_std_upperBd = max(round(obsInfo.noise_std_upperBd,2),0.001);
end
nsr               = obsInfo.noise_ratio;
noise_std         = nsr *noise_std_upperBd;  obsInfo.noise_std = noise_std;% noise added to fx_val
fx_val            = fx_val + noise_std.*randn(size(fx_val));

fprintf('u smooth: %i,   integrator: %s,  noise-std: %2.4f \n ', 1-jump_disc,integrator,noise_std); 

dx        = obsInfo.x_mesh_dx; 
data_str  = [obsInfo.example_type,kernelInfo.kernel_type,obsInfo.u_str,obsInfo.x_mesh_str,sprintf('NSR%1.1f_dx%1.4f_',nsr,dx)];

% get boundary width and x-mesh in use. --------                            
bdry_width = obsInfo.bdry_width;  % boundary space for inetration range with kernel
r_seq      = dx*(1:bdry_width);
Index_xi_inLoss = (bdry_width+1): (length(obsInfo.u_xmesh) - bdry_width); % index that x_i of u(x_i) in use for L_phi[u]

%% 2. pre-process data, get all data elements for regression:   ****** a key step significantly reducing computational cost

normalizeOn    = ~strcmp(obsInfo.example_type, 'classicalReg');
fun_g_vec      = obsInfo.fun_g_vec;
regressionData = getData4regression_auto(ux_val,fx_val,dx,obsInfo,bdry_width,Index_xi_inLoss,r_seq,data_str,normalizeOn);
% clear ux_val fx_val;

% select using rho_val = uniform, rho_L1, or rho_L2
rho_type = 'rho_L1';   % rho_L1, or rho_L2, uniform
switch  rho_type 
    case 'uniform'
       regressionData.rho_val = regressionData.rho_val0;  
    case 'rho_L1'
       regressionData.rho_val = regressionData.rho_val1; 
    case 'rho_L2'
       regressionData.rho_val = regressionData.rho_val2; 
end 


rho_val = regressionData.rho_val; 
ind_rho = find(rho_val>0);  rho_val = rho_val(ind_rho);
r_seq   = regressionData.r_seq(ind_rho);         % when r_seq is non-uniform, use dr = r_seq(2:end) - r_seq(1:end-1).       
dr      = r_seq(2)-r_seq(1);   

K_true   = kernelInfo.K_true; 
K_true_val  = K_true(r_seq);

% sanity check
sanity_checkOn = 1; 
if sanity_checkOn ==1
    g_ukxj  = regressionData.g_ukxj;
    fu_all  = regressionData.fx_vec;
    Rphiu   = 0*fu_all;
    for k=1:N
        guk   = squeeze(regressionData.g_ukxj(:,:,k));
        Rphiu(k,:) = K_true_val*guk*dx;
    end
    diff = Rphiu - fu_all;
    figure(150); clf;  subplot(311); plot(diff'); title('check: Rphi[u]-f =0 ');
end


g = regressionData.g_ukxj;
[ns, J, n0] = size(g);
k  = n0*J;
g1 = zeros(ns, k);

fx_vec = regressionData.fx_vec';    % Jxn0
f      = fx_vec(:); 
fx_vec_test = 0*fx_vec;   % % % %  fx_vec ~  convl_gu *phi_vec* dr 
for i = 1:n0
    temp  = g(:,:,i) ; 
    g1(:,(i-1)*J+1:i*J) = g(:,:,i);
    fx_vec_test(:,i) = K_true_val*temp; 
end


%%% 3. regularizations: dgcv-Tikh, compared with GCV-Tikh----------
m = 9;   % n0/m should be an integer
[x_reg1, reg_param1] = Tikh_dgcv(regressionData,'auto',m,plotOn);
[x_reg2, reg_param2] = Tikh_dgcv(regressionData,'gauss',m,plotOn,0.1);


method = 'Tikh'; 
[x_reg3,~,~,reg_param3] = Tikh_auto_basis(regressionData, 'auto-RKHS','gcv',plotOn);
[x_reg4,~,~,reg_param4] = Tikh_auto_basis(regressionData, 'Gaussian-RKHS','gcv',plotOn,0.1);

Tikhonov_err.autoRKHS_gcv  = norm(x_reg-xx) / nx;

% relative error
er1 = norm(x_reg1-xx) / nx;
er2 = norm(x_reg2-xx) / nx;
er3 = norm(x_reg3-xx) / nx;
er4 = norm(x_reg4-xx) / nx;


%%% Compare estimators  ------------------------------------------------------------
% relative errors 
fprintf('Relative L2rho Errors: \n '); 
methods_all= ["auto-RKHS-dGCV" ; "auto-RKHS-GCV"; "Guassian-dGCV"; "Gaussian-GCV"]; 
rel_err    = [er1; er2; er3; er4]; 
Tikh_lamb  = [reg_param1;reg_param2;reg_param3;reg_param4];
rel_err_fmt = compose('%.4f', rel_err);
Tikh_lamb_fmt = compose('%.4f', Tikh_lamb);
relative_err = table(methods_all,rel_err_fmt,Tikh_lamb_fmt); 
disp(relative_err)


% plot estimators 
figure(11);  
plot(r_seq,K_true_val,'k:','Linewidth', 3); 
hold on;
plot(r_seq, x_reg1, '-.','Linewidth', 2);
hold on;
plot(r_seq, x_reg3, '-.','Linewidth', 2);
hold on;
plot(r_seq, x_reg2, '-.','Linewidth', 2);
hold on;
plot(r_seq, x_reg4, '-.','Linewidth', 2);
legend('True', 'auto-dGCV','auto-GCV','Gaussian-dGCV','Gaussian-GCV');
title('Tikhonv Estimators'); 
