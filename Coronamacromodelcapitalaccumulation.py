# import packages used
import numpy as np
import tools_Exercise_1_6 as tools
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import time as time
from interpolation import interp

def setup():
    class par: pass
    # Model parameters
    par.beta = 0.999
    par.upsillon=20
    par.Lt = 1
    par.W = 20
    par.G= 0.99
    par.chi = 0
    par.xi1 = 0
    par.xi2 = 0 
    par.D = 0.005
    par.d = 0.005
    par.varphi = 0
    par.kappa1 = 1
    par.kappa2 = 8
    par.Upsillon = 0.501*par.upsillon
    par.Z = 75000
    par.gamma1 = 0.05
    par.tests = 0.01
    par.varsigma = 13
    par.varrho = 2
    par.t = 1.8
    par.phi1 = 0.2*0.37
    par.phi2 = 0.2*0.33
    par.phi3 = 0.2*0.3
    par.sigma = 0.001
    par.varrho = 0.4 
    par.alpha=0.3
    par.rho=5
    par.g=40
    par.mu = 2
    par.H = 0.65
    par.B = 0.91
    par.b=par.B
    par.delta = 0.0001
    par.varepsilon = 0.15
    par.epsilon = 0.97
    par.cash = 0
    par.alt = 0
    # Shock parameters
    par.num_M = 6
    par.M_max = 0.25
    par.num_shocks = 8
    par.num_MM = 6
    par.max_M = 5000
    
    # Convergens settings
    par.max_iter = 7000   # maximum number of iterations
    par.tol = 10e-2

    # Simulation parameters
    par.simN = 720
    par.I_ini = 0.01
    par.Q_ini = 0.00
    par.R_ini = 0.00
    par.lw_ini = 1
    par.M_ini=500

    # Setup grid
    setup_grids(par)
    return par

def setup_grids(par):
 
    #Grid of disease parameters
    par.grid_I = tools.nonlinspace(1.0e-10,par.M_max,par.num_M,1.2) # non-linear spaced points: like np.linspace with unequal spacing
    par.grid_Q = tools.nonlinspace(1.0e-10,par.M_max,par.num_M,1.2) # non-linear spaced points: like np.linspace with unequal spacing
    par.grid_R = tools.nonlinspace(1.0e-10,0.8,par.num_M,1) # non-linear spaced points: like np.linspace with unequal spacing
    par.grid_M = tools.nonlinspace(5,3000,par.num_MM,1)
    par.grid_lw = tools.nonlinspace(1.0e-10,1,100,1)
    #Gauss-Hermite
   # x,w = tools.gauss_hermite(par.num_shocks)
   # par.eps = np.exp(par.sigma*np.sqrt(2)*x)
   #  par.eps_w = w/np.sqrt(np.pi)
    return par

def solve_cons_inf(par):


    # Initalize
    class sol: pass
    sol.V = np.ones([par.num_M, par.num_M, par.num_M, par.num_MM])*0
    sol.lw = np.ones([par.num_M, par.num_M, par.num_M, par.num_MM])

    sol.it = 0   #Number of iteration
    sol.delta = 1000.0 #Different between V+ and V
    sol.S=[]
    sol.lo=[]
    sol.s=[]
    sol.wi=[]
    sol.Y=[]
    sol.i=[]
    sol.l=[]
    sol.gamma2=[]
    sol.gamma3=[]
    sol.I_plus=[]
    sol.Q_plus=[]
    sol.R_plus=[]
    sol.M_plus=[]
    sol.p=[]
    sol.pi=[]
    prcompo = 0
        #precomp
    for  I in (par.grid_I):  
        for Q in (par.grid_Q):
            for R in (par.grid_R):
                for M in (par.grid_M):
                    for lw in (par.grid_lw):
                        if lw+Q+par.D*R > 1:
                            break
                        S=(1-I-Q-R)
                        lo=(1 - lw - Q - par.D*R)
                        s=max((lw-(1-par.D)*R)*(1-I/(S+I)),0)
                        wi=max((lw-(1-par.D)*R)*(I/(S+I)),0)
                        #w=(lw+Q+lo*par.G)*par.W
                        w=par.W
                        #J=par.G*(1-par.alpha)*par.varsigma*(par.Z*par.phi2*(min(1-R-Q,0.01)/par.alpha*par.varsigma))**(1/(par.alpha-1))
                        ###K=min(par.H*(((par.delta/J)**(1/(1-par.alpha )) - I*I-par.varphi*R )/ ((2-par.alpha)*((par.upsillon*lw+par.Upsillon*lo)**(2*par.B))))**(1/(1-2*par.B)),M)
                        p_prov = (1-par.alpha)*par.varsigma*((par.Z*par.phi2*I*max(1-R-Q, 0.001)/(par.alpha*par.varsigma))**(1/(par.alpha-1)))**par.alpha
                        K=min(max((par.delta/(par.H*(1-par.b)*par.varepsilon**(1-par.b)*(par.upsillon*lw+par.Upsillon*lo)**(par.b))*p_prov)**(1/-par.b),1),M)
                        Y=max(par.H*(par.upsillon*lw+par.Upsillon*lo)**(par.b)*((par.varepsilon*K)**(1-par.b)), 1.0e-8)
                        l=((par.Z*par.phi2*I*max(1-R-Q,0.1)/(par.alpha*par.varsigma))**(1/(par.alpha-1)))*Y

                        if l>1:
                            l=1
                        if l<0:
                            l=0
                        p=(1-par.alpha)*par.varsigma*l**par.alpha*Y**(-par.alpha)
                        if l<=0:
                            p=0
                            l=0
                            Y=0
                            K=0
                        elif l>=1:
                            if p*Y>w+par.g:
                                K=min(max(((w+par.g)/((1-par.alpha)*par.varsigma*par.H*((par.upsillon*lw+par.Upsillon*lo)**(par.b-par.alpha*par.b)*par.varepsilon**((1-par.b)*(1-par.alpha)))))**(1/((1-par.b)*(1-par.alpha))),1),M)
                                Y=max(par.H*(par.upsillon*lw+par.Upsillon*lo)**(par.b)*(par.varepsilon*K)**(1-par.b), 1.0e-8)
                                p=(w+par.g)/Y
                                l=((par.Z*par.phi2*I*max(1-R-Q,0.1)/(par.alpha*par.varsigma))**(1/(par.alpha-1)))*Y
                                if l>1:
                                    l=1
                                if l<0:
                                    l=0
                            else:
                                l=1
                                K=min(max((par.delta/(par.H**(1-par.alpha)*(1-par.b)*par.varepsilon**(1+par.alpha*par.b-par.b-par.alpha)*(par.upsillon*lw+par.Upsillon*lo)**(par.b-par.alpha*par.b)*(1+par.alpha*par.alpha-par.alpha*2)))**(1/(par.b*par.alpha-par.alpha-par.b)),1),M) 
                                Y=max(par.H*(par.upsillon*lw+par.Upsillon*lo)**(par.b)*(par.varepsilon*K)**(1-par.b), 1.0e-8)
                                p=((1-par.alpha)*par.varsigma)*Y**-par.alpha
                                l=((par.Z*par.phi2*I*max(1-R-Q,0.1)/(par.alpha*par.varsigma))**(1/(par.alpha-1)))*Y
                                if l>1:
                                    l=1 
                                if l<0:
                                    l=0
                        else:
                            if p*Y>w+par.g:
                                K=min(max(((w+par.g)/(p_prov*par.H*((par.upsillon*lw+par.Upsillon*lo)**par.b)*(par.varepsilon)**(1-par.b)))**(1/(1-par.b)),1),M)
                                Y=max(par.H*(par.upsillon*lw+par.Upsillon*lo)**(par.b)*(par.varepsilon*K)**(1-par.b), 1.0e-8)
                                p=(w+par.g)/Y
                                l=((par.Z*par.phi2*I*max(1-R-Q,0.001)/(par.alpha*par.varsigma))**(1/(par.alpha-1)))*Y
                                if l>1:
                                    l=1
                                if l<0:
                                    l=0
                        if par.alt ==1:
                            if Y*p -  par.delta*K - par.xi1*I**2 - par.xi2*par.d*R < - (lw+Q)*par.W - lo*par.G*par.W:
                                Y=0
                                K=0
                                p=0
                                l=0
                                sol.pi.append(- (lw+Q)*par.W - lo*par.G*par.W)
                                wi=0
                                s=0
                        #print(p)
                        #print(l)
                        gamma2=np.array(par.sigma + par.t*par.tests/((1 + I*par.rho)**par.mu))
                        gamma3=np.array(par.gamma1 * (1+ par.kappa1/(1+Q**(1/par.kappa2))))
                        sol.I_plus.append(max(min((1-par.gamma1-gamma2)*I + par.phi1*s*wi + par.phi2*S*I*l*l + par.phi3*S*I,1),1.0e-9))
                        sol.Q_plus.append(max(min((1- gamma3)*Q + gamma2*I,1),1.0e-9))
                        sol.R_plus.append(max(min(R + par.gamma1*I + gamma3*Q,1),1.0e-9))
                        #print(Y)
                        #print(p)
                        if par.alt==1:
                            if  Y*p -  par.delta*K - par.xi1*I**2 - par.xi2*par.d*R > - (lw+Q)*par.W - lo*par.G*par.W:
                                sol.pi.append(Y*p - (lw+Q)*par.W - par.delta*K - lo*par.G*par.W - par.xi1*I**2 - par.xi2*par.d*R, 0)
                        if par.alt==0:
                            sol.pi.append(max(Y*p - (lw+Q)*par.W - par.delta*K - lo*par.G*par.W - par.xi1*I**2 - lw*par.xi2*par.d*R,0))
                        #print(Y*p - (lw+Q)*par.W - par.delta*K - lo*par.G*par.W - par.xi1*I**2 - par.xi2*par.d*R)
                        sol.M_plus.append(max(((M+Y*p - (lw+Q)*par.W - par.delta*K - lo*par.G*par.W - par.xi1*I**2 - par.xi2*par.d*R)*par.epsilon), 5))
                        prcompo +=1
                        #print(Y*p)
                        #rint(Y)
                        #print(p)
                        #print(Y*p)
                        #print(Y*p)
                        #print(K)
    #print(sol.M_plus)
    #points=np.meshgrid(par.grid_I, par.grid_Q, par.grid_R, copy=False, indexing='xy')
    points = np.asarray(([par.grid_I, par.grid_Q, par.grid_R, par.grid_M]))
    #print(np.shape(points))
    #print(min(sol.I_plus))
    #print(np.sum(sol.Q_plus))
    #print(min(sol.Q_plus))
    #print(max(sol.R_plus))
    #print(np.sum(sol.R_plus))
    #print(np.sum(sol.M_plus))
    point = np.asarray(np.transpose(np.array([sol.I_plus, sol.Q_plus, sol.R_plus, sol.M_plus])))
    #print(point)
    while (sol.delta >= par.tol and sol.it < par.max_iter):
        V_next = sol.V.copy()
        V_plus = interpolate.interpn(points, V_next, point, method='linear', bounds_error=False, fill_value=None)
        print(np.sum(V_plus))
        #print(sum(sol.pi-V_plus))
        #V_plus=interp(np.asarray(sol.I_plus), np.asarray(sol.Q_plus), np.asarray(sol.R_plus), np.asarray(sol.M_plus), np.asarray(V_next), np.asarray(point))
        ind = -1 
        # find V
        Ih = -1
        Qh = -1
        Rh = -1        
        Mh = -1
        for I in (par.grid_I): 
            Ih +=1
            for Q in (par.grid_Q):
                Qh +=1
                for R in (par.grid_R):
                    Rh +=1
                    for M in (par.grid_M):
                        Mh +=1
                        for lw in (par.grid_lw):
                            if lw+Q+par.D*R > 1:
                                break
                            ind += 1
                            V_guess =sol.pi[ind] + par.beta*V_plus[ind]
                                ##sol.V[Ih, Qh, Rh, Mh] = V_guess
                                #sol.lw[Ih, Qh, Rh, Mh]=lw
                            if V_guess > sol.V[Ih, Qh, Rh, Mh]:
                                sol.V[Ih, Qh, Rh, Mh]=V_guess
                                sol.lw[Ih, Qh, Rh, Mh]=lw
                    Mh=-1  
                Rh=-1
            Qh=-1



        # opdate delta and it
            
        sol.it += 1
        c_new = np.ravel(sol.V)
        c_old = np.ravel(V_next)
        #sol.delta = max(abs(sol.V - V_next))
        sol.delta = max(abs(c_new - c_old))
        print(sol.delta)
    return(sol)
def simu(par, sol):
    class simu: pass

    simu.S=np.zeros([par.simN])
    simu.lo=np.zeros([par.simN])
    simu.s=np.zeros([par.simN])
    simu.wi=np.zeros([par.simN])
    simu.Y=np.zeros([par.simN])
    simu.l=np.zeros([par.simN])
    simu.p=np.zeros([par.simN])
    simu.gamma2=np.zeros([par.simN])
    simu.gamma3=np.zeros([par.simN])
    simu.pi=np.zeros([par.simN])
    simu.util=np.zeros([par.simN])
    simu.K=np.zeros([par.simN])
    simu.w=np.zeros([par.simN])
    simu.c=np.zeros([par.simN])
    simu.Pos=np.zeros([par.simN])
    simu.I=np.zeros([par.simN+1])
    simu.Q=np.zeros([par.simN+1])
    simu.R=np.zeros([par.simN+1])
    simu.M=np.zeros([par.simN+1])
    simu.I[0]=(par.I_ini)
    simu.Q[0]=(par.Q_ini)
    simu.R[0]=(par.R_ini)
    simu.M[0]=(par.M_ini)
    simu.lw = np.zeros([par.simN])
    ite=0
    points = (par.grid_I, par.grid_Q, par.grid_R, par.grid_M)
    while ite < par.simN:
    #Start of simulation.
        #point=np.asarray([simu.I[ite], simu.Q[ite], simu.R[ite], simu.M[ite]])


        simu.lw[ite]=interpolate.interpn(points, sol.lw, ([simu.I[ite], simu.Q[ite], simu.R[ite], simu.M[ite]]), method='linear', bounds_error=False, fill_value=None)
        simu.lw[ite]=min(simu.lw[ite], 1-simu.Q[ite]-simu.R[ite]*par.d)
        simu.S[ite]=(1-simu.I[ite]-simu.Q[ite]-simu.R[ite])

        simu.lo[ite]=(1 - simu.lw[ite] - simu.Q[ite] - par.D*simu.R[ite])

        simu.s[ite]=(max((simu.lw[ite]-(1-par.D)*simu.R[ite])*(1-simu.I[ite]/(simu.S[ite]+simu.I[ite])),0))

        simu.wi[ite]=(max((simu.lw[ite]-(1-par.D)*simu.R[ite])*(simu.I[ite]/(simu.S[ite]+simu.I[ite])),0))

        simu.w[ite]=par.W

        p_prov = (1-par.alpha)*par.varsigma*((par.Z*par.phi2*simu.I[ite]*max(1-simu.R[ite]-simu.Q[ite], 0.01)/(par.alpha*par.varsigma))**(1/(par.alpha-1)))**par.alpha

        simu.K[ite]=min(max((par.delta/(par.H*(1-par.b)*par.varepsilon**(1-par.b)*(par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b))*p_prov)**(1/-par.b),1e-9),simu.M[ite])
        simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
        simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
        if simu.l[ite] > 1:
            simu.l[ite]=1
        if simu.l[ite] < 0:
            simu.l[ite] = 0

        simu.p[ite]=(1-par.alpha)*par.varsigma*simu.l[ite]**(par.alpha) * simu.Y[ite]**(-par.alpha)
        #print(simu.Y[ite])

        if  simu.l[ite] < 0:
            simu.l[ite]=0
            simu.p[ite] = 0
            simu.K[ite] = 0
            simu.Y[ite] = 0

        elif simu.l[ite]>=1:
            if simu.p[ite]*simu.Y[ite]>simu.w[ite]+par.g:
                simu.K[ite]=min(max(((simu.w[ite]+par.g)/((1-par.alpha)*par.varsigma*par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*par.varepsilon**((1-par.b)*(1-par.alpha)))))**(1/((1-par.b)*(1-par.alpha))),0),simu.M[ite])
                #print(simu.K[ite]
                simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                print(simu.K[ite])
                #print(simu.Y[ite]) 
                simu.p[ite]=(simu.w[ite]+par.g)/simu.Y[ite]
                simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
                if simu.l[ite] > 1:
                    simu.l[ite]=1
                if simu.l[ite] < 0:
                    simu.l[ite] = 0
                print(1)
                          
            else:
                simu.K[ite]=min(max((par.delta/(par.H**(1-par.alpha)*(1-par.b)*par.varepsilon**(1+par.alpha*par.b-par.b-par.alpha)*(par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*(1+par.alpha*par.alpha-2*par.alpha)))**(1/(par.alpha*par.b-par.b-par.alpha)),0),simu.M[ite]) 
                #simu.K[ite]=min(max((par.delta/(par.H**(1-par.alpha)*(1-par.b)*par.varepsilon**(1+par.alpha*par.b-par.b-par.alpha)*(par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*(1-par.alpha*par.alpha)))**(1/(par.alpha*par.b-par.b-par.alpha)),0),simu.M[ite])
                simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                simu.p[ite]=((1-par.alpha)*par.varsigma)/(simu.Y[ite]**par.alpha)
                simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
                print(2)
              
                if simu.p[ite]*simu.Y[ite]>simu.w[ite]+par.g:

                    simu.K[ite]=min(max(((simu.w[ite]+par.g)/((1-par.alpha)*par.varsigma*par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*par.varepsilon**((1-par.b)*(1-par.alpha)))))**(1/((1-par.b)*(1-par.alpha))),0),simu.M[ite])
                    #print(simu.K[ite]
                    simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                    print(simu.K[ite])
                    #print(simu.Y[ite]) 
                    simu.p[ite]=(simu.w[ite]+par.g)/simu.Y[ite]
                    simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
                    print(3)
                if simu.l[ite] > 1:
                    simu.l[ite]=1
                if simu.l[ite] < 0:
                    simu.l[ite] = 0
        else:
            if simu.p[ite]*simu.Y[ite]>simu.w[ite]+par.g:
                simu.K[ite]=min(max((((simu.w[ite]+par.g)/((1-par.alpha)*par.varsigma*par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*par.varepsilon**((1-par.b)*(1-par.alpha))))))**(1/((1-par.b)*(1-par.alpha))),0),simu.M[ite])
                simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                simu.p[ite]=(simu.w[ite]+par.g)/simu.Y[ite]
                simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
            
                if simu.l[ite] > 1:
                    simu.l[ite]=1
                if simu.l[ite] < 0:
                    simu.l[ite] = 0
                print(4)
        

        simu.gamma2[ite]=(np.array(par.sigma + par.t*par.tests/(1 + simu.I[ite]*par.rho)**par.mu))

        simu.gamma3[ite]=(np.array(par.gamma1 * (1+ par.kappa1/(1+simu.Q[ite]**(1/par.kappa2)))))

        simu.pi[ite]=simu.Y[ite]*simu.p[ite] -simu.K[ite]*par.delta -(simu.lw[ite]+simu.Q[ite])*par.W - simu.lo[ite]*par.G*par.W - par.xi1*simu.I[ite]**2 - par.xi2*par.d*simu.R[ite]
        if par.alt==1:
            if  simu.pi[ite] < - (simu.lw[ite]+simu.Q[ite])*par.W - simu.lo[ite]*par.G*par.W:

                simu.pi[ite] = - (simu.lw[ite]+simu.Q[ite])*par.W - simu.lo[ite]*par.G*par.W
                simu.Y[ite] = 0
                simu.K[ite] = 0
                simu.p[ite] = 0
                simu.l[ite] = 0
                simu.s[ite] = 0
                simu.wi[ite] =0

        simu.util[ite]=(par.varsigma*simu.l[ite]**par.alpha*simu.Y[ite]**(1-par.alpha)+simu.w[ite]+par.g-simu.p[ite]*simu.Y[ite]-par.Z*par.phi2*simu.I[ite]*simu.l[ite]*(1-simu.R[ite]-simu.Q[ite])- par.Z*par.phi3*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite]))

        simu.c[ite]=simu.w[ite]+par.g-simu.p[ite]*simu.Y[ite]

        simu.Pos[ite]=(par.t*par.tests/(1 + simu.I[ite]*par.rho)**par.mu)*simu.I[ite]/(par.tests)*100



        simu.I[ite+1]=(max(min((1-par.gamma1-simu.gamma2[ite])*simu.I[ite] + par.phi1*simu.s[ite]*simu.wi[ite] + par.phi2*simu.S[ite]*simu.I[ite]*simu.l[ite]*simu.l[ite] + par.phi3*simu.S[ite]*simu.I[ite],1),1.0e-9))

        simu.Q[ite+1]=(max(min((1- simu.gamma3[ite])*simu.Q[ite] + simu.gamma2[ite]*simu.I[ite],1),1.0e-9))

        simu.R[ite+1]=(max(min(simu.R[ite] + par.gamma1*simu.I[ite] + simu.gamma3[ite]*simu.Q[ite],1),1.0e-9))
        simu.M[ite+1]=max((simu.M[ite]+simu.pi[ite])*par.epsilon,1)
        if par.cash ==1 and ite==120:
            simu.M[ite+1]=max((simu.M[ite]+simu.pi[ite])*par.epsilon,1)+1000
        ite+=1

    simu.grid = np.linspace(0,ite,ite)
    simu.I = simu.I[0:ite]
    simu.Q = simu.Q[0:ite]
    simu.R = simu.R[0:ite]
    simu.M = simu.M[0:ite]
    simu.GDP = simu.p*simu.Y
    return(simu)


def simu_wl(par, sol):
    class simu: pass

    simu.S=np.zeros([par.simN])
    simu.lo=np.zeros([par.simN])
    simu.s=np.zeros([par.simN])
    simu.wi=np.zeros([par.simN])
    simu.Y=np.zeros([par.simN])
    simu.l=np.zeros([par.simN])
    simu.p=np.zeros([par.simN])
    simu.gamma2=np.zeros([par.simN])
    simu.gamma3=np.zeros([par.simN])
    simu.pi=np.zeros([par.simN])
    simu.util=np.zeros([par.simN])
    simu.K=np.zeros([par.simN])
    simu.w=np.zeros([par.simN])
    simu.c=np.zeros([par.simN])
    simu.Pos=np.zeros([par.simN])
    simu.I=np.zeros([par.simN+1])
    simu.Q=np.zeros([par.simN+1])
    simu.R=np.zeros([par.simN+1])
    simu.M=np.zeros([par.simN+1])
    simu.I[0]=(par.I_ini)
    simu.Q[0]=(par.Q_ini)
    simu.R[0]=(par.R_ini)
    simu.M[0]=(par.M_ini)
    simu.lw = np.zeros([par.simN])
    ite=0
    points = (par.grid_I, par.grid_Q, par.grid_R, par.grid_M)
    shutdown=0
    while ite < par.simN:
    #Start of simulation.
        #point=np.asarray([simu.I[ite], simu.Q[ite], simu.R[ite], simu.M[ite]])
        simu.lw[ite]=interpolate.interpn(points, sol.lw, ([simu.I[ite], simu.Q[ite], simu.R[ite], simu.M[ite]]), method='linear', bounds_error=False, fill_value=None)
        simu.lw[ite]=min(simu.lw[ite], 1-simu.Q[ite]-simu.R[ite]*par.d)
        if simu.I[ite] > 0.02:
            shutdown = 1
        if simu.I[ite] < 0.009:
            shutdown = 0
        if shutdown ==1:
            simu.lw[ite]=min(min(simu.lw[ite], 1-simu.Q[ite]-simu.R[ite]*par.d),0.25)
        
        simu.S[ite]=(1-simu.I[ite]-simu.Q[ite]-simu.R[ite])

        simu.lo[ite]=(1 - simu.lw[ite] - simu.Q[ite] - par.D*simu.R[ite])

        simu.s[ite]=(max((simu.lw[ite]-(1-par.D)*simu.R[ite])*(1-simu.I[ite]/(simu.S[ite]+simu.I[ite])),0))

        simu.wi[ite]=(max((simu.lw[ite]-(1-par.D)*simu.R[ite])*(simu.I[ite]/(simu.S[ite]+simu.I[ite])),0))

        simu.w[ite]=par.W

        p_prov = (1-par.alpha)*par.varsigma*((par.Z*par.phi2*simu.I[ite]*max(1-simu.R[ite]-simu.Q[ite], 0.01)/(par.alpha*par.varsigma))**(1/(par.alpha-1)))**par.alpha

        simu.K[ite]=min(max((par.delta/(par.H*(1-par.b)*par.varepsilon**(1-par.b)*(par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b))*p_prov)**(1/-par.b),1e-9),simu.M[ite])
        simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
        simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
        if simu.l[ite] > 1:
            simu.l[ite]=1
        if simu.l[ite] < 0:
            simu.l[ite] = 0

        simu.p[ite]=(1-par.alpha)*par.varsigma*simu.l[ite]**(par.alpha) * simu.Y[ite]**(-par.alpha)
        #print(simu.Y[ite])

        if  simu.l[ite] < 0:
            simu.l[ite]=0
            simu.p[ite] = 0
            simu.K[ite] = 0
            simu.Y[ite] = 0

        elif simu.l[ite]>=1:
            if simu.p[ite]*simu.Y[ite]>simu.w[ite]+par.g:
                simu.K[ite]=min(max(((simu.w[ite]+par.g)/((1-par.alpha)*par.varsigma*par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*par.varepsilon**((1-par.b)*(1-par.alpha)))))**(1/((1-par.b)*(1-par.alpha))),0),simu.M[ite])
                #print(simu.K[ite]
                simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                #print(simu.Y[ite]) 
                simu.p[ite]=(simu.w[ite]+par.g)/simu.Y[ite]
                simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
                if simu.l[ite] > 1:
                    simu.l[ite]=1
                if simu.l[ite] < 0:
                    simu.l[ite] = 0
                
            else:
                simu.K[ite]=min(max((par.delta/(par.H**(1-par.alpha)*(1-par.b)*par.varepsilon**(1+par.alpha*par.b-par.b-par.alpha)*(par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*(1+par.alpha*par.alpha-2*par.alpha)))**(1/(par.alpha*par.b-par.b-par.alpha)),0),simu.M[ite])
                simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                simu.p[ite]=((1-par.alpha)*par.varsigma)/(simu.Y[ite]**par.alpha)
                simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
               
                if simu.p[ite]*simu.Y[ite]>simu.w[ite]+par.g:

                    simu.K[ite]=min(max(((simu.w[ite]+par.g)/((1-par.alpha)*par.varsigma*par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*par.varepsilon**((1-par.b)*(1-par.alpha)))))**(1/((1-par.b)*(1-par.alpha))),0),simu.M[ite])
                    simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                    simu.p[ite]=(simu.w[ite]+par.g)/simu.Y[ite]
                    simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
                
                if simu.l[ite] > 1:
                    simu.l[ite]=1
                if simu.l[ite] < 0:
                    simu.l[ite]=0
        else:
            if simu.p[ite]*simu.Y[ite]>simu.w[ite]+par.g:
                simu.K[ite]=min(max(((simu.w[ite]+par.g)/((1-par.alpha)*par.varsigma*par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*par.varepsilon**((1-par.b)*(1-par.alpha)))))**(1/((1-par.b)*(1-par.alpha))),0),simu.M[ite])
                simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                simu.p[ite]=(simu.w[ite]+par.g)/simu.Y[ite]
                simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
                
                if simu.l[ite] > 1:
                    simu.l[ite]=1
                if simu.l[ite] < 0:
                    simu.l[ite] = 0

        

        simu.gamma2[ite]=(np.array(par.sigma + par.t*par.tests/(1 + simu.I[ite]*par.rho)**par.mu))

        simu.gamma3[ite]=(np.array(par.gamma1 * (1+ par.kappa1/(1+simu.Q[ite]**(1/par.kappa2)))))

        simu.pi[ite]=simu.Y[ite]*simu.p[ite] -simu.K[ite]*par.delta -(simu.lw[ite]+simu.Q[ite])*par.W - simu.lo[ite]*par.G*par.W - par.xi1*simu.I[ite]**2 - par.xi2*par.d*simu.R[ite]

        simu.util[ite]=(par.varsigma*simu.l[ite]**par.alpha*simu.Y[ite]**(1-par.alpha)+simu.w[ite]+par.g-simu.p[ite]*simu.Y[ite]-par.Z*par.phi2*simu.I[ite]*simu.l[ite]*(1-simu.R[ite]-simu.Q[ite])- par.Z*par.phi3*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite]))

        simu.c[ite]=simu.w[ite]+par.g-simu.p[ite]*simu.Y[ite]

        simu.Pos[ite]=(par.t*par.tests/(1 + simu.I[ite]*par.rho)**par.mu)*simu.I[ite]/(par.tests)*100

        simu.I[ite+1]=(max(min((1-par.gamma1-simu.gamma2[ite])*simu.I[ite] + par.phi1*simu.s[ite]*simu.wi[ite] + par.phi2*simu.S[ite]*simu.I[ite]*simu.l[ite]*simu.l[ite] + par.phi3*simu.S[ite]*simu.I[ite],1),1.0e-9))

        simu.Q[ite+1]=(max(min((1- simu.gamma3[ite])*simu.Q[ite] + simu.gamma2[ite]*simu.I[ite],1),1.0e-9))

        simu.R[ite+1]=(max(min(simu.R[ite] + par.gamma1*simu.I[ite] + simu.gamma3[ite]*simu.Q[ite],1),1.0e-9))
        simu.M[ite+1]=max((simu.M[ite]+simu.pi[ite])*par.epsilon,1)
        ite+=1

    simu.grid = np.linspace(0,ite,ite)
    simu.I = simu.I[0:ite]
    simu.Q = simu.Q[0:ite]
    simu.R = simu.R[0:ite]
    simu.M = simu.M[0:ite]
    simu.GDP = simu.p*simu.Y
    return(simu)

def simu_sl(par, sol):
    class simu: pass

    simu.S=np.zeros([par.simN])
    simu.lo=np.zeros([par.simN])
    simu.s=np.zeros([par.simN])
    simu.wi=np.zeros([par.simN])
    simu.Y=np.zeros([par.simN])
    simu.l=np.zeros([par.simN])
    simu.p=np.zeros([par.simN])
    simu.gamma2=np.zeros([par.simN])
    simu.gamma3=np.zeros([par.simN])
    simu.pi=np.zeros([par.simN])
    simu.util=np.zeros([par.simN])
    simu.K=np.zeros([par.simN])
    simu.w=np.zeros([par.simN])
    simu.c=np.zeros([par.simN])
    simu.Pos=np.zeros([par.simN])
    simu.I=np.zeros([par.simN+1])
    simu.Q=np.zeros([par.simN+1])
    simu.R=np.zeros([par.simN+1])
    simu.M=np.zeros([par.simN+1])
    simu.I[0]=(par.I_ini)
    simu.Q[0]=(par.Q_ini)
    simu.R[0]=(par.R_ini)
    simu.M[0]=(par.M_ini)
    simu.lw = np.zeros([par.simN])
    ite=0
    points = (par.grid_I, par.grid_Q, par.grid_R, par.grid_M)
    shutdown=0
    while ite < par.simN:
    #Start of simulation.
        #point=np.asarray([simu.I[ite], simu.Q[ite], simu.R[ite], simu.M[ite]])
        simu.lw[ite]=interpolate.interpn(points, sol.lw, ([simu.I[ite], simu.Q[ite], simu.R[ite], simu.M[ite]]), method='linear', bounds_error=False, fill_value=None)
        simu.lw[ite]=min(simu.lw[ite], 1-simu.Q[ite]-simu.R[ite]*par.d)
        simu.S[ite]=(1-simu.I[ite]-simu.Q[ite]-simu.R[ite])
        if simu.I[ite] > 0.02:
            shutdown = 1
        if simu.I[ite] < 0.009:
            shutdown = 0
        simu.lo[ite]=(1 - simu.lw[ite] - simu.Q[ite] - par.D*simu.R[ite])

        simu.s[ite]=(max((simu.lw[ite]-(1-par.D)*simu.R[ite])*(1-simu.I[ite]/(simu.S[ite]+simu.I[ite])),0))

        simu.wi[ite]=(max((simu.lw[ite]-(1-par.D)*simu.R[ite])*(simu.I[ite]/(simu.S[ite]+simu.I[ite])),0))

        simu.w[ite]=par.W

        p_prov = (1-par.alpha)*par.varsigma*(par.Z*par.phi2*simu.I[ite]*(max(1-simu.R[ite]-simu.Q[ite], 0.01)/(par.alpha*par.varsigma))**(1/(par.alpha-1)))**par.alpha

        simu.K[ite]=min(max((par.delta/(par.H*(1-par.b)*par.varepsilon**(1-par.b)*(par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b))*p_prov)**(1/-par.b),1e-9),simu.M[ite])
        simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
        simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
        if shutdown==1:
            simu.l[ite]=min(simu.l[ite],0.1)

        if simu.l[ite] > 1:
            simu.l[ite]=1
        if simu.l[ite] < 0:
            simu.l[ite] = 0


        simu.p[ite]=(1-par.alpha)*par.varsigma*simu.l[ite]**(par.alpha) * simu.Y[ite]**(-par.alpha)
        #print(simu.Y[ite])

        if  simu.l[ite] < 0:
            simu.l[ite]=0
            simu.p[ite] = 0
            simu.K[ite] = 0
            simu.Y[ite] = 0

        elif simu.l[ite]>=1:
            if simu.p[ite]*simu.Y[ite]>simu.w[ite]+par.g:
                simu.K[ite]=min(max(((simu.w[ite]+par.g)/((1-par.alpha)*par.varsigma*par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*par.varepsilon**((1-par.b)*(1-par.alpha)))))**(1/((1-par.b)*(1-par.alpha))),0),simu.M[ite])
                #print(simu.K[ite]
                simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                #print(simu.Y[ite]) 
                simu.p[ite]=(simu.w[ite]+par.g)/simu.Y[ite]
                simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
                if simu.l[ite] > 1:
                    simu.l[ite]=1
                if simu.l[ite] < 0:
                    simu.l[ite] = 0
                if shutdown==1:
                    simu.l[ite]=min(simu.l[ite],0.1)
                
            else:
                simu.K[ite]=min(max((par.delta/(par.H**(1-par.alpha)*(1-par.b)*par.varepsilon**(1+par.alpha*par.b-par.b-par.alpha)*(par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*(1+par.alpha*par.alpha-2*par.alpha)))**(1/(par.alpha*par.b-par.b-par.alpha)),0),simu.M[ite])
                simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                simu.p[ite]=((1-par.alpha)*par.varsigma)/(simu.Y[ite]**par.alpha)
                simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
              
                if simu.p[ite]*simu.Y[ite]>simu.w[ite]+par.g:

                    simu.K[ite]=min(max(((simu.w[ite]+par.g)/((1-par.alpha)*par.varsigma*par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*par.varepsilon**((1-par.b)*(1-par.alpha)))))**(1/((1-par.b)*(1-par.alpha))),0),simu.M[ite])
                    simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                    simu.p[ite]=(simu.w[ite]+par.g)/simu.Y[ite]
                    simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
             
                if simu.l[ite] > 1:
                    simu.l[ite]=1
                if simu.l[ite] < 0:
                    simu.l[ite]=0
                if shutdown==1:
                    simu.l[ite]=min(simu.l[ite],0.1)
        else:
            if simu.p[ite]*simu.Y[ite]>simu.w[ite]+par.g:
                simu.K[ite]=min(max(((simu.w[ite]+par.g)/((1-par.alpha)*par.varsigma*par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*par.varepsilon**((1-par.b)*(1-par.alpha)))))**(1/((1-par.b)*(1-par.alpha))),0),simu.M[ite])
                simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                simu.p[ite]=(simu.w[ite]+par.g)/simu.Y[ite]
                simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
               
                if simu.l[ite] > 1:
                    simu.l[ite]=1
                if simu.l[ite] < 0:
                    simu.l[ite] = 0
                if shutdown==1:
                    simu.l[ite]=min(simu.l[ite],0.1)
        

        simu.gamma2[ite]=(np.array(par.sigma + par.t*par.tests/(1 + simu.I[ite]*par.rho)**par.mu))

        simu.gamma3[ite]=(np.array(par.gamma1 * (1+ par.kappa1/(1+simu.Q[ite]**(1/par.kappa2)))))

        simu.pi[ite]=simu.Y[ite]*simu.p[ite] -simu.K[ite]*par.delta -(simu.lw[ite]+simu.Q[ite])*par.W - simu.lo[ite]*par.G*par.W - par.xi1*simu.I[ite]**2 - par.xi2*par.d*simu.R[ite]

        simu.util[ite]=(par.varsigma*simu.l[ite]**par.alpha*simu.Y[ite]**(1-par.alpha)+simu.w[ite]+par.g-simu.p[ite]*simu.Y[ite]-par.Z*par.phi2*simu.I[ite]*simu.l[ite]*(1-simu.R[ite]-simu.Q[ite])- par.Z*par.phi3*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite]))

        simu.c[ite]=simu.w[ite]+par.g-simu.p[ite]*simu.Y[ite]

        simu.Pos[ite]=(par.t*par.tests/(1 + simu.I[ite]*par.rho)**par.mu)*simu.I[ite]/(par.tests)*100

        simu.I[ite+1]=(max(min((1-par.gamma1-simu.gamma2[ite])*simu.I[ite] + par.phi1*simu.s[ite]*simu.wi[ite] + par.phi2*simu.S[ite]*simu.I[ite]*simu.l[ite]*simu.l[ite] + par.phi3*simu.S[ite]*simu.I[ite],1),1.0e-9))

        simu.Q[ite+1]=(max(min((1- simu.gamma3[ite])*simu.Q[ite] + simu.gamma2[ite]*simu.I[ite],1),1.0e-9))

        simu.R[ite+1]=(max(min(simu.R[ite] + par.gamma1*simu.I[ite] + simu.gamma3[ite]*simu.Q[ite],1),1.0e-9))
        simu.M[ite+1]=max((simu.M[ite]+simu.pi[ite])*par.epsilon,1)
        ite+=1

    simu.grid = np.linspace(0,ite,ite)
    simu.I = simu.I[0:ite]
    simu.Q = simu.Q[0:ite]
    simu.R = simu.R[0:ite]
    simu.M = simu.M[0:ite]
    simu.GDP = simu.p*simu.Y
    return(simu)

def simu_swl(par, sol):
    class simu: pass

    simu.S=np.zeros([par.simN])
    simu.lo=np.zeros([par.simN])
    simu.s=np.zeros([par.simN])
    simu.wi=np.zeros([par.simN])
    simu.Y=np.zeros([par.simN])
    simu.l=np.zeros([par.simN])
    simu.p=np.zeros([par.simN])
    simu.gamma2=np.zeros([par.simN])
    simu.gamma3=np.zeros([par.simN])
    simu.pi=np.zeros([par.simN])
    simu.util=np.zeros([par.simN])
    simu.K=np.zeros([par.simN])
    simu.w=np.zeros([par.simN])
    simu.c=np.zeros([par.simN])
    simu.Pos=np.zeros([par.simN])
    simu.I=np.zeros([par.simN+1])
    simu.Q=np.zeros([par.simN+1])
    simu.R=np.zeros([par.simN+1])
    simu.M=np.zeros([par.simN+1])
    simu.I[0]=(par.I_ini)
    simu.Q[0]=(par.Q_ini)
    simu.R[0]=(par.R_ini)
    simu.M[0]=(par.M_ini)
    simu.lw = np.zeros([par.simN])
    ite=0
    points = (par.grid_I, par.grid_Q, par.grid_R, par.grid_M)
    shutdown=0
    while ite < par.simN:
    #Start of simulation.
        #point=np.asarray([simu.I[ite], simu.Q[ite], simu.R[ite], simu.M[ite]])
        simu.lw[ite]=interpolate.interpn(points, sol.lw, ([simu.I[ite], simu.Q[ite], simu.R[ite], simu.M[ite]]), method='linear', bounds_error=False, fill_value=None)
        simu.lw[ite]=min(simu.lw[ite], 1-simu.Q[ite]-simu.R[ite]*par.d)
        simu.S[ite]=(1-simu.I[ite]-simu.Q[ite]-simu.R[ite])
        if simu.I[ite] > 0.02:
            shutdown = 1
        if simu.I[ite] < 0.009:
            shutdown = 0
        if shutdown ==1:
            simu.lw[ite]=min(min(simu.lw[ite], 1-simu.Q[ite]-simu.R[ite]*par.d),0.25)
        
        simu.lo[ite]=(1 - simu.lw[ite] - simu.Q[ite] - par.D*simu.R[ite])

        simu.s[ite]=(max((simu.lw[ite]-(1-par.D)*simu.R[ite])*(1-simu.I[ite]/(simu.S[ite]+simu.I[ite])),0))

        simu.wi[ite]=(max((simu.lw[ite]-(1-par.D)*simu.R[ite])*(simu.I[ite]/(simu.S[ite]+simu.I[ite])),0))

        simu.w[ite]=par.W

        p_prov = (1-par.alpha)*par.varsigma*(par.Z*par.phi2*simu.I[ite]*(max(1-simu.R[ite]-simu.Q[ite], 0.01)/(par.alpha*par.varsigma))**(1/(par.alpha-1)))**par.alpha

        simu.K[ite]=min(max((par.delta/(par.H*(1-par.b)*par.varepsilon**(1-par.b)*(par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b))*p_prov)**(1/-par.b),1e-9),simu.M[ite])
        simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
        simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
        if shutdown==1:
            simu.l[ite]=min(simu.l[ite],0.1)

        if simu.l[ite] > 1:
            simu.l[ite]=1
        if simu.l[ite] < 0:
            simu.l[ite] = 0


        simu.p[ite]=(1-par.alpha)*par.varsigma*simu.l[ite]**(par.alpha) * simu.Y[ite]**(-par.alpha)
        #print(simu.Y[ite])

        if  simu.l[ite] < 0:
            simu.l[ite]=0
            simu.p[ite] = 0
            simu.K[ite] = 0
            simu.Y[ite] = 0

        elif simu.l[ite]>=1:
            if simu.p[ite]*simu.Y[ite]>simu.w[ite]+par.g:
                simu.K[ite]=min(max(((simu.w[ite]+par.g)/((1-par.alpha)*par.varsigma*par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*par.varepsilon**((1-par.b)*(1-par.alpha)))))**(1/((1-par.b)*(1-par.alpha))),0),simu.M[ite])
                #print(simu.K[ite]
                simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                #print(simu.Y[ite]) 
                simu.p[ite]=(simu.w[ite]+par.g)/simu.Y[ite]
                simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
                if simu.l[ite] > 1:
                    simu.l[ite]=1
                if simu.l[ite] < 0:
                    simu.l[ite] = 0
                if shutdown==1:
                    simu.l[ite]=min(simu.l[ite],0.1)
                
            else:
                simu.K[ite]=min(max((par.delta/(par.H**(1-par.alpha)*(1-par.b)*par.varepsilon**(1+par.alpha*par.b-par.b-par.alpha)*(par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*(1+par.alpha*par.alpha-2*par.alpha)))**(1/(par.alpha*par.b-par.b-par.alpha)),0),simu.M[ite])
                simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                simu.p[ite]=((1-par.alpha)*par.varsigma)/(simu.Y[ite]**par.alpha)
                simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
              
                if simu.p[ite]*simu.Y[ite]>simu.w[ite]+par.g:

                    simu.K[ite]=min(max(((simu.w[ite]+par.g)/((1-par.alpha)*par.varsigma*par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*par.varepsilon**((1-par.b)*(1-par.alpha)))))**(1/((1-par.b)*(1-par.alpha))),0),simu.M[ite])
                    simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                    simu.p[ite]=(simu.w[ite]+par.g)/simu.Y[ite]
                    simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
             
                if simu.l[ite] > 1:
                    simu.l[ite]=1
                if simu.l[ite] < 0:
                    simu.l[ite]=0
                if shutdown==1:
                    simu.l[ite]=min(simu.l[ite],0.1)
        else:
            if simu.p[ite]*simu.Y[ite]>simu.w[ite]+par.g:
                simu.K[ite]=min(max(((simu.w[ite]+par.g)/((1-par.alpha)*par.varsigma*par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**(par.b-par.alpha*par.b)*par.varepsilon**((1-par.b)*(1-par.alpha)))))**(1/((1-par.b)*(1-par.alpha))),0),simu.M[ite])
                simu.Y[ite]=max(par.H*((par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])**par.b)*((par.varepsilon*simu.K[ite])**(1-par.b)), 1.0e-8)
                simu.p[ite]=(simu.w[ite]+par.g)/simu.Y[ite]
                simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
               
                if simu.l[ite] > 1:
                    simu.l[ite]=1
                if simu.l[ite] < 0:
                    simu.l[ite] = 0
                if shutdown==1:
                    simu.l[ite]=min(simu.l[ite],0.1)
        

        simu.gamma2[ite]=(np.array(par.sigma + par.t*par.tests/(1 + simu.I[ite]*par.rho)**par.mu))

        simu.gamma3[ite]=(np.array(par.gamma1 * (1+ par.kappa1/(1+simu.Q[ite]**(1/par.kappa2)))))

        simu.pi[ite]=simu.Y[ite]*simu.p[ite] -simu.K[ite]*par.delta -(simu.lw[ite]+simu.Q[ite])*par.W - simu.lo[ite]*par.G*par.W - par.xi1*simu.I[ite]**2 - par.xi2*par.d*simu.R[ite]

        simu.util[ite]=(par.varsigma*simu.l[ite]**par.alpha*simu.Y[ite]**(1-par.alpha)+simu.w[ite]+par.g-simu.p[ite]*simu.Y[ite]-par.Z*par.phi2*simu.I[ite]*simu.l[ite]*(1-simu.R[ite]-simu.Q[ite])- par.Z*par.phi3*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite]))

        simu.c[ite]=simu.w[ite]+par.g-simu.p[ite]*simu.Y[ite]

        simu.Pos[ite]=(par.t*par.tests/(1 + simu.I[ite]*par.rho)**par.mu)*simu.I[ite]/(par.tests)*100

        simu.I[ite+1]=(max(min((1-par.gamma1-simu.gamma2[ite])*simu.I[ite] + par.phi1*simu.s[ite]*simu.wi[ite] + par.phi2*simu.S[ite]*simu.I[ite]*simu.l[ite]*simu.l[ite] + par.phi3*simu.S[ite]*simu.I[ite],1),1.0e-9))

        simu.Q[ite+1]=(max(min((1- simu.gamma3[ite])*simu.Q[ite] + simu.gamma2[ite]*simu.I[ite],1),1.0e-9))

        simu.R[ite+1]=(max(min(simu.R[ite] + par.gamma1*simu.I[ite] + simu.gamma3[ite]*simu.Q[ite],1),1.0e-9))
        simu.M[ite+1]=max((simu.M[ite]+simu.pi[ite])*par.epsilon,1)
        ite+=1

    simu.grid = np.linspace(0,ite,ite)
    simu.I = simu.I[0:ite]
    simu.Q = simu.Q[0:ite]
    simu.R = simu.R[0:ite]
    simu.M = simu.M[0:ite]
    simu.GDP = simu.p*simu.Y
    return(simu)

def discount(par):
    class disc: pass
    disc.disc = np.zeros([par.simN])
    ite=0
    while ite < par.simN:
        disc.disc[ite]=par.beta**(ite+1)
        ite=ite+1
    return(disc)