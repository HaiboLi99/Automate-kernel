
function [x_reg, reg_param] = Tikh_dgcv(regressionData,rkhs_type,m,plotOn,varargin)
% compute the regularized solution by divide and conquerGCV
%
% Inputs:    
%   regressionData: generated data
%   rkhs_type: 'auto-RKHS', 'auto-no-rho', 'Gaussian-RKHS'
%   m: number of partitions
%   plotOn: plot the L-curve or GCV curve
%   varargin: bandwighth of Gussian kernel 
%
% Outputs:
%   x_reg: regularized solution (values at {s_l} points)
%   reg_param: regularization parameter estimated by dGCV
% 

if nargin < 2
    error('Not Enough Inputs')
end

switch rkhs_type
    case 'gauss'
        l = varargin{1};
    case {'auto','auto-no-rho'}
        % pass
    otherwise 
        error('Wrong RKHS kernel type')
end 

r_seq   = regressionData.r_seq;       
dx      = r_seq(2)-r_seq(1); 
ds      = dx;
rho_val = regressionData.rho_val;

g = regressionData.g_ukxj;
[ns, J, n0] = size(g);
k  = n0*J;
g1 = zeros(ns, k);

for i = 1:n0
    g1(:,(i-1)*J+1:i*J) = g(:,:,i);
end

g1 = g1';  % n0J x ns
L = g1 * diag(1./rho_val) / sqrt(n0) * sqrt(dx);   

if strcmp(rkhs_type, 'auto')
    G      = regressionData.G;             
    Gbar_D = G./(rho_val*rho_val');       
elseif strcmp(rkhs_type, 'auto-no-rho')
    Gbar_D = regressionData.G; 
elseif strcmp(rkhs_type, 'gauss')
    G_fun = @(s1,s2) Gauss(s1, s2, l);
    r_seq = r_seq(:);    % nsx1
    rr1   = r_seq * ones(1,ns);
    rr2   = rr1';
    G_mat = arrayfun(G_fun, rr1(:), rr2(:));
    Gbar_D = reshape(G_mat, ns, ns);    
end

basis_D = g1*Gbar_D*ds;     % auto-basis from the Representer theorem: (n0J,ns)x(ns,ns)-->(n0J,ns)
Xi = basis_D * ds;
% Sigma_D = g1 * basis_D' * ds;

[N, ~] = size(g1);
if mod(N,m) ~= 0
    error('N should be exactly devided by m')
end

l = N / m;

S = zeros(l, m);
U = zeros(l, l, m);

for i = 1:m 
    ind = (i-1)*l+1:i*l;
    Sigma_ii = g1(ind,:) * Xi(:,ind)';
    [~,s,V] = csvd(Sigma_ii);
    S(:,i) = s;
    U(:,:,i) = V;
end

[reg_param, G, param_list] = dgcv(f, g1, Xi, S, U);

ns = size(G_D, 2);
G1 = G_D';    %  nsxN
x_reg = zeros(ns, 1);
tol = 1e-16;
for i = 1:m 
    fi   = f((i-1)*l+1:i*l);
    G_Di = G1(:, (i-1)*l+1:i*l);
    Ui = U(:,:,i);
    Si = S(:,i);
    ind = find(Si>tol);
    S1i = 1 ./ (Si(ind)+reg_param);
    x_reg = x_reg + G_Di * (Ui(:,ind) * (diag(S1i) * (Ui(:,ind)' * fi)));
end

x_reg = x_reg / m;

end


%%--------------------------------------------------------
function [reg_min,G,reg_param] = dgcv(f, g, Xi, S, U)
% Plot the dGCV function and find its minimum.

npoints = 200;                       % Number of points on the curve.
smin_ratio = 100*eps;                % Smallest regularization parameter.

% Vector of regularization parameters
reg_param = zeros(npoints,1); 
G = reg_param; 
[l, ~] = size(S);
reg_param(npoints) = max([S(l,1),S(1,1)*smin_ratio]);
ratio = (S(1,1)/reg_param(npoints))^(1/(npoints-1));
for i=npoints-1:-1:1
    reg_param(i) = ratio*reg_param(i+1); 
end

% Vector of GCV-function values
for i=1:npoints
    G(i) = dgcvfun(reg_param(i), f, g, Xi, S, U);
end 

% Plot dGCV function.
figure;
loglog(reg_param,G,'-'), xlabel('\lambda'), ylabel('dGCV(\lambda)');
title('dGCV function');

% Find minimum, if requested.
[~,minGi] = min(G);    % Initial guess.
gfun = @(x) dgcvfun(x, f, g, Xi, S, U);
reg_min = fminbnd(gfun, reg_param(min(minGi+1,npoints)),reg_param(max(minGi-1,1)),optimset('Display','off')); 
minG = gfun(reg_min);  % Minimum of dGCV function.

ax = axis;
HoldState = ishold; hold on;
loglog(reg_min,minG,'*r',[reg_min,reg_min],[minG/1000,minG],':r')
title(['dGCV function, minimum at \lambda = ',num2str(reg_min)])
axis(ax)
if (~HoldState)
    hold off;
end

end


%-------------------------------------------------
function G = dgcvfun(lambda, f, g, Xi, S, U)
% Function of the divide and conquer GCV.
%
% Inputs:    
%   lambda: regularization parameter
%   f: observation, Nx1, N=n_0*J
%   g, Xi: used to form Sigma
%   S: store the eigenvalues of {Sigma_D_i}_{i=1}^{m};
%      S is lxm, where each Sigma_D_i are lxl, and l*m=N.
%   U: store the eigenvectors of {Sigma_D_i}_{i=1}^{m};
%      U is lxlxm, where lxl store the eigenvectors.
%
% Outputs:
%   G: function value of dgcvfun

% Check for input arguments
[N, ~] = size(f);
if size(g,1) ~= N 
    error('Dimension not consistent')
end
[l, m] = size(S);
if l*m ~= N 
    error('Dimension not consistent')
end
if (size(U,1)~=size(U,2)) || (size(U,1)~=l) || (size(U,3)~=m)
    error('Dimension not consistent')
end  

% compute Sigma_lambda * f, and the trace term
val2 = 0;    % value of the trace term
tol = 1e-16;
f1 = zeros(N,1);
for i = 1:m 
    fi = f((i-1)*l+1:i*l);
    Ui = U(:,:,i);
    Si = S(:,i);
    ind = find(Si>tol);
    r = length(ind);
    S1i = zeros(l,1);
    S1i(1:r) = 1 ./ (Si(ind)+lambda);
    S1i(r+1:l) = zeros(l-r,1);

    f1((i-1)*l+1:i*l) = Ui * (diag(S1i)*(Ui'*fi));

    val2 = val2 + sum(Si(ind)./(Si(ind)+lambda));
end

% compute 1/m*Sigma_D*Sigma_lambda*f
f_hat = 1/m * g * (Xi * f1);

% residual norm
val1 = norm(f-f_hat)^2;

% compute the dGCV value
G = val1 / ((N-1/m*val2))^2;

end
