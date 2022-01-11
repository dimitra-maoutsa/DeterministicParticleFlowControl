# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 04:08:30 2022

@author: maout
"""


import torch
import numpy as np
import logging
import torched_score_function_multid_seperate_all_dims
import time

class torched_DPFC(object):
    """
    Deterministic particle flow control top-level class implemented in pytorch.

    Provides the necessary functions to sample the required probability
    flows and estimate the controls.

    Attributes
    ----------
    t1 : float
        Initial time.
    t2: float
        end time point.
    y1: array_like
        initial position.
    y2: array_like
        terminal position.
    f: function, callable
        drift function handle.
    g: float or array_like
        diffusion coefficient or function handle.
    N: int
        number of particles/trajectories.
    M: int
        number of sparse points for grad log density estimation.
    reweight: boolean
        determines if reweighting will follow.
    U: function, callable
        reweighting function to be employed during reweighting,
        dimensions :math:`dim_y1,t \\to 1`.
    dens_est: str
              - 'nonparametric' : non parametric density estimation (this was
                                  used in the paper)
              - TO BE ADDED:
                  - 'hermit1' : parametic density estimation empoying hermite
                            polynomials (physiscist's)
                 - 'hermit2' : parametic density estimation empoying hermite
                            polynomials (probabilists's)
                 - 'poly' : parametic density estimation empoying simple polynomials
                 - 'RBF' : parametric density estimation employing radial basis functions.
    kern: str
        type of kernel: 'RBF' or 'periodic' (only the 'RBF' was used and gives
                        robust results. Do not use 'periodic' yet!).
    reject: boolean
        parameter indicating whether non valid backward trajectories will be rejected.
    plotting: boolean
        parameter indicating whether bridge statistics will be plotted.
    f_true: funtion, callable
        in case of Brownian bridge reweighting this is the true forward drift
        for simulating the forward dynamics.
    brown_bridge: boolean,
        determines if the reweighting concearns contstraint or reweighting with
        respect to brownian bridge.
    deterministic: boolean,
        indicates the type of dynamics the particles will follow.
        If False the flows are simulated with stochastic path sampling.
    device: string,
        indicates the device where computations will be exacuted.
        `cpu` or `gpu/cuda` or `tpu` if available.

    Methods
    -------
    forward_sampling_Otto:
        Creates samples of the forward flow.
    forward sampling():
        Samples the forward flow with stochatic particle trajectories.
    f_seperate(x,t):
        Drift for the deterministic propagation of partcles that are at time t
        in position x.
    backward_simulation():
        Sampling the backward density with stochastic particles.
    reject_trajectories():
        Rejects backward trajectories that do not end up in the vicinity of the
        initial point.
        Run only if the instance is attribute "reject" is set to True.
        Gives logging.warning messages.
    forward_sampling_Otto_true():
        Relevant only when forward sampling happens with Brownian bridge.
    """


    def __init__(self, t1, t2, y1, y2, f, g, N, M, reweight=False, U=None,
                 dens_est='nonparametric', reject=True, kern='RBF',
                 f_true=None, brown_bridge=False, deterministic=True,
                 device=None):

        self.device = device
        # dimensionality of the system
        self.dim = torch.tensor(y1.size, dtype=torch.float32, device=self.device)  # dimensionality of the system
        self.t1 = t1
        self.t2 = t2
        if not torch.is_tensor(y1):
            self.y1 = torch.tensor(y1, dtype=torch.float32, device=self.device)
            self.y2 = torch.tensor(y2, dtype=torch.float32, device=self.device)
        else:
            self.y1 = y1
            self.y2 = y2


        ##density estimation stuff
        self.kern = kern
        if kern == 'periodic':
            self.kern = 'RBF'
            logging.warning('Please do not use periodic kernel yet!')
            logging.warning('For all the numerical experiments RBF was used')
            logging.warning('We changed your choice to RBF')
        # DRIFT /DIFFUSION
        self.f = f
        self.g = torch.tensor(g, dtype=torch.float32, device=self.device) #scalar or array

        ### PARTICLE DISCRETISATION
        self.N = torch.tensor(N, dtype=torch.float32, device=self.device)

        self.N_sparse = M

        self.dt = torch.tensor(0.001, dtype=torch.float32, device=self.device)
                               #((t2-t1)/k)
        ### reject unreasonable backward trajectories that do not return
        ### to initial condition
        self.reject = reject
        ### indicator for what type of dynamics the particles follow
        self.deterministic = deterministic


        self.timegrid = torch.arange(self.t1, self.t2+self.dt/2, self.dt,
                                     dtype=torch.float32, device=self.device)
        self.k = self.timegrid.size
        ### reweighting
        self.brown_bridge = brown_bridge
        self.reweight = reweight
        if self.reweight:
            self.U = U
            if self.brown_bridge:
                #storage for forward trajectories with true drift
                self.Ztr = torch.zeros(self.dim, self.N, self.k,
                                       dtype=torch.float32, device=self.device)
                self.f_true = f_true



        #storage for forward trajectories
        self.Z = torch.zeros(self.dim, self.N, self.k, dtype=torch.float32,
                             device=self.device)
        #storage for backward trajectories
        self.B = torch.zeros(self.dim, self.N, self.k, dtype=torch.float32,
                             device=self.device )
        self.ln_roD = [] ## storing the estimated forward logarithmic gradients


        ##the stochastic sampling is provided for comparison
        if self.deterministic:
            self.forward_sampling_Otto()
            ### if a Brownian bridge is used for forward sampling
            if self.reweight and self.brown_bridge:
                self.forward_sampling_Otto_true()
        else:
            self.forward_sampling()
        ## the backward function selects internally for type of dynamics
        self.backward_simulation()
        # if self.reject:
        #     self.reject_trajectories()


    def forward_sampling(self):
        """
        Sampling forward probability flow with stochastic particle dynamics.
        If reweighting is required at every time step the particles are
        appropriatelly reweighted accordint to function :math:`U(x,t)`

        Returns
        -------
        int
            Returns 0 to make sure everything runs correctly.
            The sampled density is stored in place in the array `self.Z`.

        """
        logging.info('Sampling forward...')
        W = torch.ones(self.N, 1, dtype=torch.float32, device=self.device)/self.N
        for ti, tt in enumerate(self.timegrid):

            if ti == 0:
                self.Z[0, :, 0] = self.y1[0]
                self.Z[1, :, 0] = self.y1[1]
            else:
                for i in range(self.N):
                    #self.Z[:,i,:] = sdeint.itoint(self.f, self.g, self.Z[i,0], self.timegrid)[:,0]
                    self.Z[:, i, ti] = self.Z[:, i, ti-1] + \
                        self.dt* self.f(self.Z[:, i, ti-1]) + \
                        (self.g)*torch.empty(self.dim,1).normal_(mean=0,std=np.sqrt(self.dt))

                ###WEIGHT
                if self.reweight == True:
                    if ti > 0:
                        W[:, 0] = torch.exp(self.U(self.Z[:, :, ti]))
                        W = W/torch.sum(W)

                        ###REWEIGHT with pot TO DO:
                        Tstar = reweight_optimal_transport_multidim(self.Z[:, :, ti].T, W)

                        self.Z[:, :, ti] = (self.Z[:, :, ti])@Tstar

        for di in range(self.dim):
            self.Z[di, :, -1] = self.y2[di]
        logging.info('Forward sampling done!')
        return 0




    ### relevant only when forward trajectories follow brownian brifge -
    ###this simulates forward trajectories with true f
    def f_seperate_true(self, x, t):
        """
        (Relevant only when forward sampling happens with Brownian bridge
        reweighting)
        Wrapper for the drift function of the deterministic particles with the
        actual f (system drift) minus the logarithmic gradient term computed
        on current particles positions.
        Provided for easy integration, and can be passed to ode integrators.

        Parameters
        ----------
        x : 2d-array,
            Particle positions (dimension x number of particles).
        t : float,
            Time t within the [t1,t2] interval.

        Returns
        -------
        2d-array
            Returns the deterministic forces required to ntegrate the particle
            positions for one time step,
            i.e. return :math:`f(x,t)-\\frac{1}{2}\\sigma^2\\nabla \\rho_t(x)`,
            evaluated at the current positions x and t.

        """
        dimi, N = x.shape
        bnds = torch.zeros(dimi, 2, dtype=torch.float32, device=self.device)
        for ii in range(dimi):
            bnds[ii] = [torch.min(x[ii, :]), torch.max(x[ii, :])]

        Sxx = torch.tensor([torch.distributions.Uniform(low=bnd[0], high=bnd[1]).sample(self.N_sparse)
                            for bnd in bnds], dtype=torch.float32,
                           device=self.device)

        #gpsi = torch.zeros(dimi, N, dtype=torch.float32, device=self.device)
        lnthsc = 2*torch.std(x, dim=1)


        gpsi = torched_score_function_multid_seperate_all_dims(torch.t(x),
                                                               torch.t(Sxx),
                                                               func_out=False,
                                                               C=0.001,
                                                               l=lnthsc,
                                                               kern=self.kern,
                                                               device=self.device)

        return self.f_true(x, t)-0.5* self.g**2* gpsi


    ### effective forward drift - estimated seperatelly for each dimension
    #plain GP prior
    def f_seperate(self, x, t):
        """
        Computes the deterministic forces for the evolution of the deterministic
        particles for the current particle positions,
        ie. drift minus the logarithmic gradient term.
        Is used as a wrapper for evolving the particles,
        and can be provided to "any" ODE integrator.

        Parameters
        ----------
        x : 2d-array,
            Particle positions (dimension x number of particles).
        t : float,
            Time t within the [t1,t2] interval.

        Returns
        -------
        2d-array
            Returns the deterministic forces required to ntegrate the particle
            positions for one time step,
            i.e. return :math:`f(x,t)-\\frac{1}{2}\\sigma^2\\nabla \\rho_t(x)`,
            evaluated at the current positions x and t.

        """


        dimi, N = x.shape
        ### detect min and max of forward flow for each dimension
        ### we want to know the state space volume of the forward flow
        bnds = torch.zeros(dimi, 2, dtype=torch.float32, device=self.device)
        for ii in range(dimi):
            bnds[ii] = [torch.min(x[ii, :]), torch.max(x[ii, :])]
        # sum_bnds = np.sum(bnds) ##this is for detecting if sth goes wrong i.e. trajectories explode
        # if np.isnan(sum_bnds) or np.isinf(sum_bnds):
        #     ##if we get unreasoble bounds just plot the first 2 dimensions of the trajectories
        #     plt.figure(figsize=(6, 4)), plt.plot(self.Z[0].T, self.Z[1].T, alpha=0.3)
        #     plt.show()

        ##these are the inducing points
        ## here we select them from a uniform distribution within the state space volume spanned from the forward flow
        Sxx = torch.tensor([torch.distributions.Uniform(low=bnd[0], high=bnd[1]).sample(self.N_sparse)
                            for bnd in bnds], dtype=torch.float32,
                           device=self.device)
        #gpsi = np.zeros((dimi, N))
        lnthsc = 2*torch.std(x, dim=1)

        gpsi = torched_score_function_multid_seperate_all_dims(torch.t(x),
                                                               torch.t(Sxx),
                                                               func_out=False,
                                                               C=0.001,
                                                               l=lnthsc,
                                                               kern=self.kern,
                                                               device=self.device)
        return self.f(x, t)-0.5* self.g**2* gpsi

     ###same as forward sampling but without reweighting - this is for bridge reweighting
        ### not for constraint reweighting
    def forward_sampling_Otto_true(self):
        """
        (Relevant only when forward sampling happens with Brownian bridge
        reweighting)
        Same as forward sampling but without reweighting.

        Returns
        -------
        int
            Returns 0 to make sure everything runs correctly.
            The sampled density is stored in place in the array `self.Ztr`.

        See also
        ---------
        DPFC.forward_sampling, DPFC.forward_sampling_Otto

        """
        logging.info('Sampling forward with deterministic particles and true drift...')
        #W = np.ones((self.N,1))/self.N
        for ti, tt in enumerate(self.timegrid):

            if ti == 0:
                for di in range(self.dim):
                    self.Ztr[di, :, 0] = self.y1[di]

            elif ti == 1: #propagate one step with stochastic to avoid the delta function
                                          #substract dt because I want the time at t-1
                self.Ztr[:, :, ti] = self.Ztr[:, :, ti-1] + \
                                      self.dt*self.f_true(self.Ztr[:, :, ti-1], tt-self.dt)+\
                                 (self.g)*torch.empty([self.dim, self.N]).normal_(mean=0, std=np.sqrt(self.dt))
            else:
                self.Ztr[:, :, ti] = self.Ztr[:, :, ti-1] + \
                    self.dt* self.f_seperate_true(self.Ztr[:, :, ti-1], tt-self.dt)

        logging.info('Forward sampling with Otto true is ready!')
        return 0



    def forward_sampling_Otto(self):
        """
        Samples the forward probability flow with deterministic particle
        dynamics.
        If required at every timestep a particle reweighting takes place
        employing the weights obtained from the exponentiated path constraint
        :math:`U(x,t)`

        Returns
        -------
        int
            Returns 0 to make sure everything runs correctly.
            The sampled density is stored in place in the array `self.Z`.

        """
        logging.info('Sampling forward with deterministic particles...')
        W = torch.ones(self.N, 1, dtype=torch.float32, device=self.device)/self.N
        for ti, tt in enumerate(self.timegrid):
            if ti == 0:
                for di in range(self.dim):
                    self.Z[di, :, 0] = self.y1[di]
                    if self.brown_bridge:
                        self.Z[di, :, -1] = self.y2[di]
                    ## we start forward trajectories for a delta function.
                    ##in principle we could start from an arbitrary distribution
                    ##if you want to start from a normal uncomen the following and comment the above initialisation for y1
                    #self.Z[di,:,0] = np.random.normal(self.y1[di], 0.05, self.N)
            elif ti == 1: #propagate one step with stochastic to avoid the delta function
                                           #substract dt because I want the time at t-1
                self.Z[:, :, ti] = self.Z[:, :, ti-1] + \
                                    self.dt*self.f(self.Z[:, :, ti-1], tt-self.dt)+\
                                 (self.g)*\
                                     torch.empty([self.dim, self.N]).normal_(mean=0, std=np.sqrt(self.dt))
            else:
                self.Z[:, :, ti] = self.Z[:, :, ti-1] +\
                    self.dt* self.f_seperate(self.Z[:, :, ti-1], tt-self.dt)
                ###REWEIGHT
            if self.reweight == True:
                if ti > 0:

                    W[:, 0] = torch.exp(self.U(self.Z[:, :, ti], tt)*self.dt) #-1
                    W = W/torch.sum(W)

                    ###REWEIGHT
                    start = time.time()
                    Tstar = reweight_optimal_transport_multidim(self.Z[:, :, ti].T, W)
                    #print(Tstar)
                    if ti == 3:
                        stop = time.time()
                        logging.info('Timepoint: %d needed '%ti, stop-start)
                    self.Z[:, :, ti] = ((self.Z[:, :, ti])@Tstar) #####
        logging.info('Forward sampling with Otto is ready!')
        return 0

    def density_estimation(self, ti, rev_ti):
        rev_t = rev_ti
        grad_ln_ro = torch.zeros(self.dim, self.N, dtype=torch.float32,
                                 device=self.device)
        lnthsc = 2*torch.std(self.Z[:, :, rev_t], dim=1)
        bnds = torch.zeros(self.dim, 2, dtype=torch.float32, device=self.device)
        for ii in range(self.dim):
            bnds[ii] = [max(np.min(self.Z[ii, :, rev_t]), np.min(self.B[ii, :, rev_ti])), min(np.max(self.Z[ii, :, rev_t]), np.max(self.B[ii, :, rev_ti]))]

        #sparse points
        Sxx = torch.tensor([torch.distributions.Uniform(low=bnd[0], high=bnd[1]).sample(self.N_sparse)
                            for bnd in bnds], dtype=torch.float32,
                           device=self.device)

        #estimate density from forward (Z) and evaluate at current postitions of backward particles (B)
        #grad_ln_ro = score_function_multid_seperate(self.Z[:, :, rev_t].T, Sxx.T, func_out=True, C=0.001, which=1, l=lnthsc, which_dim=di+1, kern=self.kern)(self.B[:, :, rev_ti].T)
        grad_ln_ro = torched_score_function_multid_seperate_all_dims(torch.t(self.Z[:, :, rev_t]),
                                                               torch.t(Sxx),
                                                               func_out=True,
                                                               C=0.001,
                                                               l=lnthsc,
                                                               kern=self.kern,
                                                               device=self.device)(torch.t(self.B[:, :, rev_ti]))

        return grad_ln_ro


    def bw_density_estimation(self, rev_ti):
        """
        Estimates the logaritmic gradient of the backward flow evaluated at
        particle positions of the backward flow.


        Parameters
        ----------

        rev_ti : int,
                 indicates the time point in the timegrid where the estimation
                 will take place, i.e. for time t=self.timegrid[rev_ti.

        Returns
        -------
        grad_ln_b: 2d-array,
                    with the logarithmic gradients of the time reversed
                    (backward) flow (dim x N) for the timestep `rev_ti`.

        """
        grad_ln_b = torch.zeros(self.dim, self.N, dtype=torch.float32,
                                device=self.device)
        lnthsc = 2*torch.std(self.B[:, :, rev_ti], dim=1)

        bnds = torch.zeros(self.dim, 2, dtype=torch.float32, device=self.device)
        for ii in range(self.dim):
            bnds[ii] = [max(np.min(self.Z[ii, :, rev_ti]), np.min(self.B[ii, :, rev_ti])), min(np.max(self.Z[ii, :, rev_ti]), np.max(self.B[ii, :, rev_ti]))]
        #sparse points
        Sxx = torch.tensor([torch.distributions.Uniform(low=bnd[0], high=bnd[1]).sample(self.N_sparse)
                            for bnd in bnds], dtype=torch.float32,
                           device=self.device)



        grad_ln_b = torched_score_function_multid_seperate_all_dims(torch.t(self.B[:, :, rev_ti]),
                                                               torch.t(Sxx),
                                                               func_out=False,
                                                               C=0.001,
                                                               l=lnthsc,
                                                               kern=self.kern,
                                                               device=self.device)
        return grad_ln_b


    def backward_simulation(self):
        """
        Sample time reversed flow with deterministic dynamics (or stochastic if
        `self.deterministic == False`).
        Trajectories are stored in place in `self.B` array of dimensionality
        (dim x N x timegrid.size).
        `self.B` does not require a timereversion at the end, everything
        is stored in the correct order.

        Returns
        -------
        int
            Returns 0 to ensure everything was executed correctly.

        """

        for ti, tt in enumerate(self.timegrid[:-1]):
            if ti == 0:
                for di in range(self.dim):
                    self.B[di, :, -1] = self.y2[di]
            else:

                rev_ti = self.k -ti-1
                #density estimation of forward particles
                grad_ln_ro = self.density_estimation(ti, rev_ti+1)

                if (ti == 1 and self.deterministic) or (not self.deterministic):

                    self.B[:, :, rev_ti] = self.B[:, :, rev_ti+1] -\
                                            self.f(self.B[:, :, rev_ti+1], self.timegrid[rev_ti+1])*self.dt + \
                                                self.dt*self.g**2*grad_ln_ro +\
                                                    (self.g)*\
                                                        torch.empty([self.dim, self.N]).normal_(mean=0, std=np.sqrt(self.dt))
                else:
                    grad_ln_b = self.bw_density_estimation(rev_ti+1)
                    self.B[:, :, rev_ti] = self.B[:, :, rev_ti+1] -\
                                          (self.f(self.B[:, :, rev_ti+1], self.timegrid[rev_ti+1])-\
                                           self.g**2*grad_ln_ro +0.5*self.g**2*grad_ln_b)*self.dt

        for di in range(self.dim):
            self.B[di, :, 0] = self.y1[di]
        return 0


    """
    def reject_trajectories(self):

        Reject backward trajectories that do not reach the vicinity of the
        initial point.
        Deletes in place relevant rows of the `self.B` array that contains
        the time reversed trajectories.

        Returns
        -------
        int
            Returns 0.


        fplus = self.y1+self.f(self.y1, self.t1)*self.dt+6*self.g**2 *np.sqrt(self.dt)
        fminus = self.y1+self.f(self.y1, self.t1) *self.dt-6*self.g**2 *np.sqrt(self.dt)
        reverse_order = np.zeros(self.dim)
        #this is an indicator if along one of the dimensions fplus
        #is smaller than fminus
        for iii in range(self.dim):
            if fplus[iii] < fminus[iii]:
                reverse_order[iii] = 1
        to_delete = np.zeros(self.N)
        ##these will be one if ith trajectory is out of bounds

        ## checking if out of bounds for each dim
        for iii in range(self.dim):
            if reverse_order[iii] == 0:
                to_delete += np.logical_not(np.logical_and(self.B[iii, :, 1] < fplus[iii], self.B[iii, :, 1] > fminus[iii]))
            elif reverse_order[iii] == 1:
                to_delete += np.logical_not(np.logical_and(self.B[iii, :, 1] > fplus[iii], self.B[iii, :, 1] < fminus[iii]))

        sinx = np.where(to_delete >= 1)[0]
        #sinx = np.where(np.logical_or(np.logical_not(np.logical_and(self.B[0, :, 1] < fplus[0], self.B[0, :, 1] > fminus[0])), np.logical_not(np.logical_and(self.B[0, :, 1] < fplus[0], self.B[0, :, 1] > fminus[0]))))[0]
                           #((self.B[1,:,-2]<fplus[1]))  ) & ( & (self.B[1,:,-2]>fminus[1]) )  ))[0]


        #logging.warning("Identified %d invalid bridge trajectories "%len(sinx))
        # if self.reject:
        #     logging.warning("Deleting invalid trajectories...")
        #     sinx = sinx[::-1]
        #     for element in sinx:
        #         self.B = np.delete(self.B, element, axis=1)
        return 0
    """


    def calculate_u(self, grid_x, ti):
        """
        Computes the control at position(s) grid_x at timestep ti
        (i.e. at time self.timegrid[ti]).

        Parameters
        ----------
        grid_x : ndarray,
                 size d x number of points to be evaluated.
        ti     : int,
                  time index in timegrid for the computation of u.


        Returns
        -------
        u_t: ndarray,
             same size as grid_x. These are the controls u(grid_x, t),
             where t=self.timegrid[ti].

        """
        #a = 0.001
        #grad_dirac = lambda x,di: - 2*(x[di] -self.y2[di])*
        #np.exp(- (1/a**2)* (x[0]- self.y2[0])**2)/(a**3 *np.sqrt(np.pi))
        u_t = torch.zeros(grid_x.T.shape, dtype=torch.float32,
                          device=self.device)


        lnthsc1 = 2*torch.std(self.B[:, :, ti], dim=1)
        lnthsc2 = 2*torch.std(self.Z[:, :, ti], dim=1)


        bnds = torch.zeros(self.dim, 2, dtype=torch.float32,
                           device=self.device)

        for ii in range(self.dim):
            if self.reweight == False or self.brown_bridge == False:
                bnds[ii] = [max(torch.min(self.Z[ii, :, ti]), torch.min(self.B[ii, :, ti])), min(torch.max(self.Z[ii, :, ti]), torch.max(self.B[ii, :, ti]))]
            else:
                bnds[ii] = [max(torch.min(self.Ztr[ii, :, ti]), torch.min(self.B[ii, :, ti])), min(torch.max(self.Ztr[ii, :, ti]), torch.max(self.B[ii, :, ti]))]

        if ti <= 5 or (ti >= self.k-5):
            if self.reweight == False or self.brown_bridge == False:
                ##for the first and last 5 timesteps, to avoid numerical singularities just assume gaussian densities
                for di in range(self.dim):
                    mutb = torch.mean(self.B[di, :, ti])
                    stdtb = torch.std(self.B[di, :, ti])
                    mutz = torch.mean(self.Z[di, :, ti])
                    stdtz = torch.std(self.Z[di, :, ti])
                    u_t[di] = -(grid_x[:, di]- mutb)/stdtb**2 - (-(grid_x[:, di]- mutz)/stdtz**2)
            elif self.reweight == True and self.brown_bridge == True:
                for di in range(self.dim):
                    mutb = torch.mean(self.B[di, :, ti])
                    stdtb = torch.std(self.B[di, :, ti])
                    mutz = torch.mean(self.Ztr[di, :, ti])
                    stdtz = torch.std(self.Ztr[di, :, ti])
                    u_t[di] = -(grid_x[:, di]- mutb)/stdtb**2 - (-(grid_x[:, di]- mutz)/stdtz**2)
        else: #if ti > 5:
            ### clipping not used at the end but provided here for cases when
            ### number of particles is small
            ### and trajectories fall out of simulated flows
            ### TO DO: add clipping as an option to be selected when
            ### initialising the function
            ### if point for evaluating control falls out of the region where we
            ### have points, clip the points to
            ### fall within the calculated region - we do not change the
            ### position of the point, only the control value will be
            ### calculated with clipped positions
            bndsb =torch.zeros(self.dim, 2, dtype=torch.float32,
                          device=self.device)
            bndsz = torch.zeros(self.dim, 2, dtype=torch.float32,
                          device=self.device)
            for di in range(self.dim):
                bndsb[di] = [torch.min(self.B[di, :, ti]), torch.max(self.B[di, :, ti])]
                bndsz[di] = [torch.min(self.Z[di, :, ti]), torch.max(self.Z[di, :, ti])]

            ## clipping not used at the end!
            ###cliping the values of points when evaluating the grad log p
            grid_b = grid_x#np.clip(grid_x, bndsb[0], bndsb[1])
            grid_z = grid_x#np.clip(grid_x, bndsz[0], bndsz[1])

            Sxx = torch.tensor([torch.distributions.Uniform(low=bnd[0], high=bnd[1]).sample(self.N_sparse)
                            for bnd in bnds], dtype=torch.float32,
                           device=self.device)

            #for di in range(self.dim):
            score_Bw = torched_score_function_multid_seperate_all_dims(torch.t(self.B[:, :, ti]),
                                                           torch.t(Sxx),
                                                           func_out=True,
                                                           C=0.001,
                                                           l=lnthsc1,
                                                           kern=self.kern,
                                                           device=self.device)(grid_b)

                #score_Bw = score_function_multid_seperate(self.B[:, :, ti].T, Sxx.T, func_out=True, C=0.001, which=1, l=lnthsc1, which_dim=di+1, kern=self.kern)(grid_b)
            if self.reweight == False or self.brown_bridge == False:
                score_Fw = torched_score_function_multid_seperate_all_dims(torch.t(self.Z[:, :, ti]),
                                                           torch.t(Sxx),
                                                           func_out=True,
                                                           C=0.001,
                                                           l=lnthsc2,
                                                           kern=self.kern,
                                                           device=self.device)(grid_z)
                #score_Fw = score_function_multid_seperate(self.Z[:, :, ti].T, Sxx.T, func_out=True, C=0.001, which=1, l=lnthsc2, which_dim=di+1, kern=self.kern)(grid_z)
            else:
                bndsztr = torch.zeros(self.dim, 2, dtype=torch.float32,
                          device=self.device)
                for ii in range(self.dim):
                    bndsztr[di] = [torch.min(self.Ztr[di, :, ti]), torch.max(self.Ztr[di, :, ti])]
                grid_ztr = grid_x #np.clip(grid_x, bndsztr[0], bndsztr[1])
                lnthsc3 = 2*torch.std(self.Ztr[:, :, ti], dim=1)
                #score_Fw = score_function_multid_seperate(self.Ztr[:, :, ti].T, Sxx.T, func_out=True, C=0.001, which=1, l=lnthsc3, which_dim=di+1, kern=self.kern)(grid_ztr)
                score_Fw = torched_score_function_multid_seperate_all_dims(torch.t(self.Ztr[:, :, ti]),
                                                           torch.t(Sxx),
                                                           func_out=True,
                                                           C=0.001,
                                                           l=lnthsc3,
                                                           kern=self.kern,
                                                           device=self.device)(grid_ztr)
            u_t = score_Bw - score_Fw

        return u_t


    def check_if_covered(self, X, ti):
        """
        Checks if test point X falls within forward and backward densities at
        timepoint timegrid[ti].


        Parameters
        ----------
        X : array 1x dim or Kxdim
            Point in state space where control is evaluated.
        ti : int
            Index in timegrid array indicating the time within the
            time interval [t1,t2].

        Returns
        -------
        Boolean variable - True if the text point X falls within the densities.

        """
        covered = True
        bnds = torch.zeros(self.dim, 2, dtype=torch.float32,
                          device=self.device)
        for ii in range(self.dim):
            bnds[ii] = [max(torch.min(self.Z[ii, :, ti]), torch.min(self.B[ii, :, ti])),
                        min(torch.max(self.Z[ii, :, ti]), torch.max(self.B[ii, :, ti]))]
            #bnds[ii] = [np.min(self.B[ii,:,ti]),np.max(self.B[ii,:,ti])]

            covered = covered * ((X[ii] >= bnds[ii][0]) and (X[ii] <= bnds[ii][1]))

        return covered
