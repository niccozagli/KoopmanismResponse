% Built from a copy of DBwell2D_master.m to be a hard-coded script that
% produces plots for paper

clear
close all
addpath(genpath(pwd))

%% INITIALIZATION
%% Initialise parameters

% Dynamics: double well potential
V = @(x,y) (x.^2 - 1).^2 + y.^2;
dV = @(x) [4.*(x(1).^3) - 4.*x(1), 2.*x(2)];
% We use these separate defs to plot vector field
dVx = @(x) 4.*x.*(x.^2 - 1);
dVy = @(y) 2.*y;

% Choose noise strength
sigma = 0.7;

% Choose initial cond. (x0,y0) randomly within defined region
x0 = -1.5 + 3*rand(1,2);

% integration parameters
% tmin = 0 is implied
tmax_tot = 5*10^5;
dt_int = 0.001;

transient = 100;
tmax = tmax_tot + transient;
num_tsteps_int = round(tmax/dt_int);

% ===EDMD Parameters===
% highest power in monomial dictionary 
Kmax = 10;
% dictionary size
kn = nchoosek(Kmax+2,Kmax);
% Choose EDMD timestep (flight time)
dt = 0.05;

% ===Discrete grids for plotting eigenfunctions, density===
% Define granularity of grid
xdiscrete = -2:0.05:2; ydiscrete = xdiscrete;
% Create grid
[xgrid,ygrid] = meshgrid(xdiscrete,ydiscrete);

% Count bins
nbins_x = length(xdiscrete)-1;nbins_y = length(ydiscrete)-1;
nbins_tot = nbins_x*nbins_y;

% Create grid at midpoints for plotting mu
xdisc_mid = zeros(nbins_x,1);
ydisc_mid = zeros(nbins_y,1);
for i=1:nbins_x
    xdisc_mid(i) = (xdiscrete(i+1) + xdiscrete(i))/2;
end
for i=1:nbins_y
    ydisc_mid(i) = (ydiscrete(i+1) + ydiscrete(i))/2;
end
[xgrid_mid,ygrid_mid] = meshgrid(xdisc_mid,ydisc_mid);

%% MAIN
%% Integrate
tic
y_full = integrate2D(x0,num_tsteps_int,dV,sigma,dt_int);
toc

% first remove transient
y_full = y_full(round(transient/dt_int)+1:end,:);

% then filter trajectory to use for EDMD
filter = round(dt/dt_int);
y = y_full(1:filter:end,:);

disp('Integration completed')

% ===Get measure probabilistically===
mu = histcounts2(y_full(:,1),y_full(:,2),xdiscrete,ydiscrete,'Normalization','pdf');
mu = mu.';
disp('Invariant density updated')

% Plot invariant density & vector field
contourf(xdisc_mid,ydisc_mid,mu,16,'LineWidth',0.05)
daspect([1 1 1])
cc = colorbar;
cc.Label.String = '$\rho_0(x,y)$';
cc.Label.Interpreter = 'latex';
cc.Label.FontSize = 11;
xlabel('$x$','Interpreter','latex')
ylabel('$y$','Interpreter','latex')
%Add vector field
hold on
x_sparse = -1.75:0.25:1.75; y_sparse = x_sparse;
[xgrid_sparse,ygrid_sparse] = meshgrid(x_sparse,y_sparse);
quiver(xgrid_sparse,ygrid_sparse,-dVx(xgrid_sparse),-dVy(ygrid_sparse),'r','LineWidth',1)
hold off

%% EDMD
tic
[G,A] = EDMD_2Dvectorised(y,Kmax);
toc
%% Run this cell for Hermitian DMD
A_h = (A+A')/2;
A = A_h;
% This produces virtually identical results to EDMD
%% 
K = pinv(G)*A;

% Get spectral properties of K
[Xi,W,lambda] = get_spectral_properties(K);
disp('EDMD completed and spectral properties calculated and updated')

% Plot eigenvalues of K
angles = linspace(0, 2*pi, 359);
xcoords=cos(angles); ycoords=sin(angles);
plot(xcoords, ycoords)
xlim([-1.1 1.1])
ylim([-1.1 1.1])
pbaspect([1 1 1])
hold on
plot(real(lambda), imag(lambda), 'o')
hold off

% eigenvalues of generator: decay rates
rates = log(lambda)/dt;

%% Run to plot decay rates
notoplot = 30;
plot(real(rates(1:notoplot)), imag(rates(1:notoplot)),'bo')
xline(0)
hold off

%% ANALYSIS
%% Plot first 15 eigenfunctions

% Define Koopman eigenfunctions on grid
PXgrid = monodict2D([xgrid(:),ygrid(:)],Kmax);
Phigrid = PXgrid*real(Xi);
% For this system, eigenvalues should be real, so we take real part to
% avoid propagating errors

% Inaccurate results in areas of low density, so we cut off edge of plot
chopoff = round(0.8/0.05);
prange = chopoff+1:length(xgrid)-chopoff;

tt = tiledlayout(3,5);

for eignum = 2:16
    Phitoplot = reshape(Phigrid(:,eignum),[length(xgrid),length(ygrid)]);
    nexttile
    contourf(xgrid(prange,prange), ygrid(prange,prange), Phitoplot(prange,prange),24,'LineWidth',0.1)
    ax=gca;
    ax.FontSize = 12;
    clim([-max(abs(Phitoplot(prange,prange)),[],"all") max(abs(Phitoplot(prange,prange)),[],"all")])
   
    % Turn on/off x,y tick labels:
    if eignum < 12
        xticklabels({})
    end
    %if mod(eignum,5) ~= 1
    %    yticklabels({})
    %end 

    title(sprintf('$\\lambda_{%d} = %0.4g$',eignum-1,rates(eignum)),'FontSize',14,'Interpreter','latex')
    pbaspect([1 1 1])
end

tt.TileSpacing = 'tight';
tt.Padding = 'tight';
xlabel(tt,'$x$','FontSize',20,'Interpreter','latex')
ylabel(tt,'$y$','FontSize',20,'Interpreter','latex')

cc = colorbar;
cc.Layout.Tile = 'east';
cc.Ticks = 0;
cc.Label.Interpreter = 'latex';
cc.Label.String = '$\varphi_j(x,y)$';
cc.Label.FontSize = 20;   

%% Correlation functions
% Generate new, independent trajectory 
tmax_2 = tmax_tot + transient;
num_tsteps_int_2 = round(tmax_2/dt_int);
x0_2 = -1.5 + 3*rand(1,2);

% Free up memory by clearing original trjactory first
clear y_full

tic
y_2 = integrate2D(x0_2,num_tsteps_int_2,dV,sigma,dt_int);
toc

dt_2 = 0.01;
filter_2 = round(dt_2/dt_int);

y_2 = y_2(round(transient/dt_int)+1:filter_2:end,:);

%% Next 2 cells are optional consistency check 
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Check (auto)correlation functions of individual eigenfunctions
% Get dictionary, eigenfunctions on (a section of) trajectory

Psi = monodict2D(y_2(10^7:2*10^7,:),Kmax);
Phi = Psi*real(Xi);
% So as not to run out of memory, we only evalute the first 30
% eigenfunctions on this trajectory.
% eigenvectors should be real for gradient system, so we enforce this to
% avoid keeping track of very small imaginary parts resulting from
% numerical errors

%% Plot autocorrelation
% Choose eigenfunction
eignum = 3;

maxlag = 400/dt_2;
tplot_max = 10;
nplot = 0:tplot_max/dt;

% First, take correlation of eigenfns stored in vectors as a fn of time numerically
signal = real(Phi(:,eignum)) - mean(real(Phi(:,eignum)));
[cf,lags] = xcorr(signal,maxlag,'biased');

% Plot numerical correlation results as blue points
plot(lags(maxlag+1:end)*dt_2,cf(maxlag+1:end)/max(cf),'.','MarkerSize',5)

hold on
% Now also plot analytic prediction given by exponential decay with rate set
% by eigenvalue as red dashed line
plot(nplot*dt, lambda(eignum).^(nplot), '--','LineWidth',2)
hold off

xlim([0 tplot_max])
xlabel('$t$','Interpreter','latex')
ylabel(sprintf('$C_{\\varphi_{%d}}(t)$',eignum),'Interpreter','latex')

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%% Reconstruct observables
% Observable list:
obs_x = @(x,y) x;
obs_y = @(x,y) y;
obs_rad = @(x,y) x.^2 + y.^2;
obs_3 = @(x,y) cos(2*x) + sin(2*y);
obs_4 = @(x,y) sin(2*x) + cos(2*y);

% Decompose polynomial observables by hand
h_x = zeros(kn,1); h_y = zeros(kn,1); h_rad = zeros(kn,1);

h_y(2) = 1;
h_x(3) = 1;
h_rad(4) = 1; h_rad(6) = 1;

v_y = W'*h_y; v_x = W'*h_x; v_rad = W'*h_rad;

% Decompose non-polynomial observables
obsnum_3 = obs_3(xgrid,ygrid);
[hnum_3, vnum_3] = decompose_genfn_numerical(xdiscrete,ydiscrete,obsnum_3,Kmax,W);

obsnum_4 = obs_4(xgrid,ygrid);
[hnum_4, vnum_4] = decompose_genfn_numerical(xdiscrete,ydiscrete,obsnum_4,Kmax,W);

%% Correlation functions of observables

% Numerical correlations
max_lag = 200/dt_2;
% Odd functions (in x)
[cf_x,lags_x] = get_cf_2D_better(y_2,obs_x,obs_x,max_lag);
[cf_4,lags_4] = get_cf_2D_better(y_2,obs_4,obs_4,max_lag);
% Even functions
[cf_y,lags_y] = get_cf_2D_better(y_2,obs_y,obs_y,max_lag);
[cf_rad,lags_rad] = get_cf_2D_better(y_2,obs_rad,obs_rad,max_lag);
[cf_3,lags_3] = get_cf_2D_better(y_2,obs_3,obs_3,max_lag);

% Koopman correlations
scalarprod_matrix= Xi'*G*Xi;
% Odd functions
Kcorrfn_x = KoopmanCorrs(v_x,v_x,lambda,scalarprod_matrix);
Kcorrfn_4 = KoopmanCorrs(vnum_4,vnum_4,lambda,scalarprod_matrix);
% Even functions
Kcorrfn_y = KoopmanCorrs(v_y,v_y,lambda,scalarprod_matrix);
Kcorrfn_rad = KoopmanCorrs(v_rad,v_rad,lambda,scalarprod_matrix);
Kcorrfn_3 = KoopmanCorrs(vnum_3,vnum_3,lambda,scalarprod_matrix);

%% Plot correlation functions of odd observables

tplot_max = 180;
res = 0.1;
nplot = 0:res:tplot_max/dt;
tplot = 0:dt*res:tplot_max;

cf_filter = 100;
cf_plot_range = max_lag+1:cf_filter:2*max_lag+1;

hold on
plot(lags_x(cf_plot_range)*dt_2,cf_x(cf_plot_range),'bo', 'MarkerFaceColor', 'b','LineWidth',1)
plot(tplot,Kcorrfn_x(nplot),'b-','LineWidth',2)

plot(lags_4(cf_plot_range)*dt_2,cf_4(cf_plot_range),'ro','LineWidth',1)
plot(tplot,Kcorrfn_4(nplot),'r-.','LineWidth',2)
hold off
xlim([0 tplot_max])
ylim([-0.01 inf])
box on

xlabel('$t$','Interpreter','latex')
ylabel('$C_{f}(t)$','Interpreter','latex')
legend('$f  = x$ - Numerical','$f  = x$ - Spectral','$f  = \sin(2x) + \cos(2y)$ - Numerical','$f  = \sin(2x) + \cos(2y)$ - Spectral','Interpreter','latex')

%% Plot correlation functions of even observables

tplot_max = 2;
res = 0.01;
nplot = 0:res:tplot_max/dt;
tplot = 0:dt*res:tplot_max;

cf_filter = 4;
cf_plot_range = max_lag+1:cf_filter:2*max_lag+1;

hold on
plot(lags_y(cf_plot_range)*dt_2,cf_y(cf_plot_range),'bo', 'MarkerFaceColor', 'b','LineWidth',1)
plot(tplot,Kcorrfn_y(nplot),'b-','LineWidth',2)

plot(lags_rad(cf_plot_range)*dt_2,cf_rad(cf_plot_range),'ro','LineWidth',1.5)
plot(tplot,Kcorrfn_rad(nplot),'r--','LineWidth',2)

plot(lags_3(cf_plot_range)*dt_2,cf_3(cf_plot_range),'mo','LineWidth',0.5)
plot(tplot,Kcorrfn_3(nplot),'m-.','LineWidth',2)
hold off
xlim([0 tplot_max])
ylim([0 inf])
box on

xlabel('$t$','Interpreter','latex')
ylabel('$C_{f}(t)$','Interpreter','latex')
legend('$f  = y$ - Numerical','$f  = y$ - Spectral','$f  = x^2 + y^2$ - Numerical','$f  = x^2 + y^2$ - Spectral','$f  = \cos(2x) + \sin(2y)$ - Numerical','$f  = \cos(2x) + \sin(2y)$ - Spectral','Interpreter','latex')

%% Decompose into individual contributions - odd observables

linetypes_b = {'b-','b:','b--','b-.','b-','b:','b--','b-.','b-','b:','b--','b-.'};
linetypes_r = {'r-','r:','r--','r-.','r-','r:','r--','r-.','r-','r:','r--','r-.'};

iter = 1;

hold on
for i=2:kn
    if abs(KoopmanCorrs_singleterm(v_x,v_x,lambda,scalarprod_matrix,i,1))>1e-3
        plot(tplot,KoopmanCorrs_singleterm(v_x,v_x,lambda,scalarprod_matrix,i,nplot),linetypes_b{iter},'LineWidth',3,'DisplayName',sprintf('$k=%d$',i-1))
        iter = iter+1;
    end
end
%iter = 1;
for i=2:kn
    if KoopmanCorrs_singleterm(vnum_4,vnum_4,lambda,scalarprod_matrix,i,1)>1e-3
        plot(tplot,KoopmanCorrs_singleterm(vnum_4,vnum_4,lambda,scalarprod_matrix,i,nplot),linetypes_r{iter},'LineWidth',2,'DisplayName',sprintf('$k=%d$',i-1))
        iter = iter+1;
    end
end

xlabel('$t$','Interpreter','latex')
ylabel('$C_{f}(t)$','Interpreter','latex')
legend('Interpreter','latex')
box on

yscale log
ylim([1e-4 1])
xlim([0 2])
hold off
%% Decompose into individual contributions - even observables

linetypes_b = {'b-','b:','b--','b-.','b-','b:','b--','b-.','b-','b:','b--','b-.'};
linetypes_r = {'r-','r:','r--','r-.','r-','r:','r--','r-.','r-','r:','r--','r-.'};
linetypes_m = {'m-','m:','m--','m-.','m-','m:','m--','m-.','m-','m:','m--','m-.','m-','m:','m--','m-.'};

iter = 1;

hold on
for i=2:kn
    if abs(KoopmanCorrs_singleterm(v_y,v_y,lambda,scalarprod_matrix,i,2))>1e-4
        plot(tplot,KoopmanCorrs_singleterm(v_y,v_y,lambda,scalarprod_matrix,i,nplot),linetypes_b{iter},'LineWidth',4,'DisplayName',sprintf('$k=%d$',i-1))
        iter = iter+1;
    end
end
iter = 1;
for i=2:kn
    if KoopmanCorrs_singleterm(v_rad,v_rad,lambda,scalarprod_matrix,i,1)>5e-3
        plot(tplot,KoopmanCorrs_singleterm(v_rad,v_rad,lambda,scalarprod_matrix,i,nplot),linetypes_r{iter},'LineWidth',3,'DisplayName',sprintf('$k=%d$',i-1))
        iter = iter+1;
    end
end
iter = 1;
for i=2:kn
    if KoopmanCorrs_singleterm(vnum_3,vnum_3,lambda,scalarprod_matrix,i,1)>5e-3
        plot(tplot,KoopmanCorrs_singleterm(vnum_3,vnum_3,lambda,scalarprod_matrix,i,nplot),linetypes_m{iter},'LineWidth',2,'DisplayName',sprintf('$k=%d$',i-1))
        iter = iter+1;
    end
end

xlabel('$t$','Interpreter','latex')
ylabel('$C_{f}(t)$','Interpreter','latex')
legend('Interpreter','latex')
box on

yscale log
ylim([1e-4 0.4])
xlim([0 2])
hold off

%% Perturbations & response
%% Write down analytic expression for measure

invdens = @(x,y) exp(-(2/sigma.^2)*(x.^4-2*x.^2+1+y.^2));
z = integral2(invdens, -10,10,-10,10);
rho = @(x,y) exp(-(2/sigma.^2)*(x.^4-2*x.^2+1+y.^2))/z;

%% Define Gamma analytically & decompose

% First perturbation
gamma_an1 = @(x,y) (2/sigma.^2)*(4*x.^3 - 4*x);
% Second perturbation
gamma_an2 = @(x,y) (32/sigma.^2)*(x.^4 - x.^2) - 4;

% decompose these functions wrt monomial dictionary
h_g1 = zeros(kn,1); 
h_g2 = zeros(kn,1); 

h_g1(3) = -8/sigma.^2;
h_g1(10) = 8/sigma.^2;
h_g2(1) = -4;
h_g2(6) = -32/sigma.^2;
h_g2(15) = 32/sigma.^2;

% Koopman modes
v_g1 = W'*h_g1; v_g2 = W'*h_g2;

% Calculate response functions for prev. defined observables to these perturbations
GreenFn_x = KoopmanCorrs(v_x,v_g1,lambda,scalarprod_matrix);
GreenFn_rad = KoopmanCorrs(v_rad,v_g2,lambda,scalarprod_matrix);

%% Numerical reponse experiments
%% First perturbation
pert1 = @(x) [1,0];

tmax_resp = 50;
dt_resp = 0.01;
num_tsteps_resp = round(tmax_resp/dt_resp);
tt_resp = 0:dt_resp:tmax_resp;

num_samples = 5*10^6;

% Choose strength of perturbation
eps = 0.005;

tic
gf_num_x_eps005_order6_dt01_to50 = get_response_numerical_single_eps(obs_x,pert1,eps,num_samples,y_2,num_tsteps_resp,dt_resp,dV,sigma);
toc

%% Plot spectral & numerical Green's functions

tplot_max = tmax_resp;

res = 1;
nplot = 0:res:tplot_max/dt;
tplot = 0:dt*res:tplot_max;

plot(tplot,GreenFn_x(nplot),'LineWidth',2)

hold on
box on 

plot(tt_resp,gf_num_x_eps005_order6_dt01_to50,'rx')

hold off
xlim([0 tplot_max])

xlabel('$t$','Interpreter','latex')
ylabel('$G_{f}(t)$','Interpreter','latex')
legend('Spectral Estimate','Numerical Simulation','Interpreter','latex')

%% Second perturbation

tmax_resp2 = 1.2;
dt_resp2 = 0.005;
num_tsteps_resp2 = round(tmax_resp2/dt_resp2);
tt_resp2 = 0:dt_resp2:tmax_resp2;

pert2 = @(x) [4.*x(:,1),zeros(size(x,1),1)];

% Perturbation experiment
num_samples = 5*10^6;
% Choose strength of perturbation
eps = 0.005;
tic
gf_num_rad_eps005_order5to6_dt005_to1_2 = get_response_numerical_single_eps(obs_rad,pert2,eps,num_samples,y_2,num_tsteps_resp2,dt_resp2,dV,sigma);
toc

%% Plot spectral & numerical Green's functions

tplot_max = tmax_resp2;

res = 0.1;
nplot = 0:res:tplot_max/dt;
tplot = 0:dt*res:tplot_max;

plot(tplot,GreenFn_rad(nplot),'LineWidth',2)

hold on

xlim([0 tplot_max])
box on 

plot(tt_resp2,gf_num_rad_eps005_order5to6_dt005_to1_2,'r.','LineWidth',2)
plot(tt_resp2,gf_num2_rad_eps5_order5_dt005_to1_2,'g^','LineWidth',2)

hold off

xlabel('$t$','Interpreter','latex')
ylabel('$G_{f}(t)$','Interpreter','latex')
legend('Spectral Estimate','Numerical Simulation','Interpreter','latex')

