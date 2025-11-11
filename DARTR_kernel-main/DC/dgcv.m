function [reg_min,G,reg_param] = dgcv(f, Sigma_D, S, U)
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
    G(i) = dgcvfun(reg_param(i), f, Sigma_D, S, U);
end 

% Plot dGCV function.
figure;
loglog(reg_param,G,'-'), xlabel('\lambda'), ylabel('dGCV(\lambda)');
title('dGCV function');

% Find minimum, if requested.
[~,minGi] = min(G);    % Initial guess.
gfun = @(x) dgcvfun(x, f, Sigma_D, S, U);
reg_min = fminbnd(gfun, reg_param(min(minGi+1,npoints)),reg_param(max(minGi-1,1)),optimset('Display','off')); 
minG = gfun(reg_min);  % Minimum of GCV function.

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
function G = dgcvfun(lambda, f, Sigma_D, S, U)
% Function of the divide and conquer GCV.
%
% Inputs:    
%   lambda: regularization parameter
%   f: observation, Nx1, N=n_0*J
%   Sigma_D: Gram matrix, NxN
%   S: store the eigenvalues of {Sigma_D_i}_{i=1}^{m};
%      S is lxm, where each Sigma_D_i are lxl, and l*m=N.
%   U: store the eigenvectors of {Sigma_D_i}_{i=1}^{m};
%      U is lxlxm, where lxl store the eigenvectors.
%
% Outputs:
%   G: function value of dgcvfun

% Check for input arguments
[N, ~] = size(f);
if size(Sigma_D,1) ~= N 
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
f_hat = 1/m * Sigma_D * f1;

% residual norm
val1 = norm(f-f_hat)^2;

% compute the dGCV value
G = val1 / ((N-1/m*val2))^2;

end
