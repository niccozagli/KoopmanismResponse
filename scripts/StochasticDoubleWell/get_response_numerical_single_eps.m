function greenfn_single_eps = get_response_numerical_single_eps(obs,pert,eps,num_samples,inv_meas,M,dt,dV,noise)
    
    % Sample the data ditributed wrt the inv. meas uniformly
    filter_for_inits = round(length(inv_meas(:,1))/num_samples);
    init_conds = inv_meas(1:filter_for_inits:end,:);
    
    % Apply the perturbation to these sampled points
    inits_pert = init_conds + eps.*pert(init_conds);

    obs_unpert = obs(inv_meas(:,1),inv_meas(:,2));
    obs_unpert_av = mean(obs_unpert);

    % Perform experiments
    obs_ens_av = average_pert_obs(obs,inits_pert,num_samples,M,dV,noise,dt);
    
    greenfn_single_eps = (obs_ens_av-obs_unpert_av)/eps;

end