# import packages used
import numpy as np
import tools_Exercise_1_6 as tools
#import scipy.optimize as optimize
import scipy.interpolate as interpolate
import time as time

def setup():
    class par: pass
    # Model parameters
    par.beta = 0.999
    par.B = 0.33
    par.upsillon=20
    par.Lt = 1
    par.W = 20
    par.G= 0.99
    par.chi = 30
    par.xi1 = 0
    par.xi2 = 0
    par.D = 0.005
    par.d = 0.005
    par.varphi = 0
    par.kappa1 = 1
    par.kappa2 = 8
    par.Upsillon = 0.51*par.upsillon
    par.Z = 75000
    par.gamma1 = 0.055
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
    par.H = 4.7
    # Shock parameters
    par.num_M = 10
    par.M_max = 0.3
    par.num_shocks = 8
    
    # Convergens settings
    par.max_iter = 100000   # maximum number of iterations
    par.tol = 10e-2

    # Simulation parameters
    par.simN = 1400
    par.I_ini = 0.01
    par.Q_ini = 0.00
    par.R_ini = 0.00
    par.lw_ini = 1

    # Setup grid
    setup_grids(par)
    return par

def setup_grids(par):
 
    #Grid of disease parameters
    par.grid_I = tools.nonlinspace(1.0e-10,par.M_max,par.num_M,1.5) # non-linear spaced points: like np.linspace with unequal spacing
    par.grid_Q = tools.nonlinspace(1.0e-10,par.M_max,par.num_M,1.5) # non-linear spaced points: like np.linspace with unequal spacing
    par.grid_R = tools.nonlinspace(1.0e-10,0.7,par.num_M,1.5) # non-linear spaced points: like np.linspace with unequal spacing
    par.grid_lw = tools.nonlinspace(1.0e-10,1,100,1)
    #Gauss-Hermite
   # x,w = tools.gauss_hermite(par.num_shocks)
   # par.eps = np.exp(par.sigma*np.sqrt(2)*x)
   #  par.eps_w = w/np.sqrt(np.pi)
    return par

def solve_cons_inf(par):


    # Initalize
    class sol: pass
    sol.V = np.ones([par.num_M, par.num_M, par.num_M])*1e-5
    sol.lw = np.zeros([par.num_M, par.num_M, par.num_M])

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
    sol.p=[]
    sol.pi=[]
    prcompo = 0
        #precomp
    for  I in (par.grid_I):  
        for Q in (par.grid_Q):
            for R in (par.grid_R):
                for lw in (par.grid_lw):
                    if lw+Q+par.D*R > 1:
                        break
                    S=(1-I-Q-R)
                    lo=(1 - lw - Q - par.D*R)
                    s=min(max((lw-(1-par.D)*R)*(1-I/(S+I)),0),1)
                    wi=min(max((lw-(1-par.D)*R)*(I/(S+I)),0),1)
                    Y=max(par.H*np.log(par.upsillon*lw+par.Upsillon*lo)-(par.chi*I)**2 - par.varphi*R, 1.0e-8)
                    w=(lw+Q+lo*par.G)*par.W
                    #print(Y)
                    l=((par.Z*par.phi2*I*max(1-R-Q,1.0e-9)/(par.alpha*par.varsigma))**(1/(par.alpha-1)))*Y
                    if l<0:
                        p=0
                        l=0
                    elif l>1:
                        l=1
                        p=((1-par.alpha)*par.varsigma)*Y**-par.alpha
                    else:
                        p=((1-par.alpha)*par.varsigma*l**(par.alpha) * Y**(-par.alpha))
                    if p*Y>w+par.g:
                        p=(w+par.g)/Y
                    #print(p)
                    #print(l)
                    gamma2=np.array(par.sigma + (par.t*par.tests/((1 + I*par.rho)**par.mu)))
                    gamma3=np.array(par.gamma1 * (1+ par.kappa1/(1+Q**(1/par.kappa2))))
                    sol.I_plus.append(max( min((1-par.gamma1-gamma2)*I + par.phi1*s*wi + par.phi2*S*I*l*l + par.phi3*S*I,1),1.0e-9))
                    sol.Q_plus.append(max(min((1- gamma3)*Q + gamma2*I,1),1.0e-9))
                    sol.R_plus.append(max(min(R + par.gamma1*I + gamma3*Q,1),1.0e-9))
                    sol.pi.append(Y*p - (lw+Q)*par.W - lo*par.G*par.W - par.xi1*I**2 - par.xi2*par.d*R)
                    #print(Y*p - (lw+Q)*par.W - lo*par.G*par.W - (par.xi1*I)**2 - par.xi2*par.d*R)
                    #print(par.W+par.g-Y*p)
                    prcompo +=1
                    
    #points=np.meshgrid(par.grid_I, par.grid_Q, par.grid_R, copy=False, indexing='xy')
    points = (par.grid_I, par.grid_Q, par.grid_R)
    #print(np.shape(points))
    #print(max(sol.I_plus))
    #print(min(sol.I_plus))
    #print(max(sol.Q_plus))
    #print(min(sol.Q_plus))
    #print(max(sol.R_plus))
    #print(min(sol.R_plus))
    point = np.transpose(np.array([sol.I_plus, sol.Q_plus, sol.R_plus]))
    while (sol.delta >= par.tol and sol.it < par.max_iter):
        V_next = sol.V.copy()
        V_plus = interpolate.interpn(points, V_next, point, method='linear', bounds_error=False, fill_value=None)
        ind = -1
        # find V
        Ih = -1
        Qh = -1
        Rh = -1
        
        for I in (par.grid_I): 
            Ih +=1
            for Q in (par.grid_Q):
                Qh +=1
                for R in (par.grid_R):
                    Rh +=1
                    for lw in (par.grid_lw):
                        if lw+Q+par.D*R > 1:
                            break
                        ind += 1
                        V_guess =sol.pi[ind] + par.beta*V_plus[ind]
                        if V_guess > sol.V[Ih, Qh, Rh]:
                            sol.V[Ih, Qh, Rh]=V_guess
                            sol.lw[Ih, Qh, Rh]=lw
                       
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
    simu.c=np.zeros([par.simN])
    simu.I=np.zeros([par.simN+1])
    simu.Q=np.zeros([par.simN+1])
    simu.R=np.zeros([par.simN+1])
    simu.w=np.zeros([par.simN])
    simu.Pos=np.zeros([par.simN])
    simu.I[0]=(par.I_ini)
    simu.Q[0]=(par.Q_ini)
    simu.R[0]=(par.R_ini)
    simu.lw =np.zeros([par.simN])
    simu.lw[0] = 1
    ite=0
    points = (par.grid_I, par.grid_Q, par.grid_R)
    while ite < par.simN:
    #Start of simulation.
        simu.lw[ite]=min(interpolate.interpn(points, sol.lw, ([simu.I[ite], simu.Q[ite], simu.R[ite]]), method='linear', bounds_error=False, fill_value=None), 1-simu.Q[ite]-simu.R[ite]*par.D)

        simu.lw[ite]=min(simu.lw[ite], 1-simu.Q[ite]-simu.R[ite]*par.d)

        if ite == 0:

            simu.lw[ite]=1

        simu.S[ite]=(1-simu.I[ite]-simu.Q[ite]-simu.R[ite])

        simu.lo[ite]=(1 - simu.lw[ite] - simu.Q[ite] - par.D*simu.R[ite])

        simu.s[ite]=(max((simu.lw[ite]-(1-par.D)*simu.R[ite])*(1-simu.I[ite]/(simu.S[ite]+simu.I[ite])),1.0e-9))

        simu.wi[ite]=(max((simu.lw[ite]-(1-par.D)*simu.R[ite])*(simu.I[ite]/(simu.S[ite]+simu.I[ite])),1.0e-9))

        simu.Y[ite]=(max(par.H*np.log(par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])-(par.chi*simu.I[ite])**2 - par.varphi*simu.R[ite], 1.0e-9))

        simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
        
        simu.w[ite]=(simu.lo[ite]*par.G+simu.lw[ite]+simu.Q[ite])*par.W

        if  simu.l[ite] < 0:
            simu.l[ite]=0
            simu.p[ite] = 0

        elif simu.l[ite]>1:
            simu.l[ite]=1
            simu.p[ite]=((1-par.alpha)*par.varsigma)/(simu.Y[ite]**par.alpha)
        else:
            simu.p[ite]=(1-par.alpha)*par.varsigma*simu.l[ite]**(par.alpha) * simu.Y[ite]**(-par.alpha)

        if simu.p[ite]*simu.Y[ite]>simu.w[ite]+par.g:
            simu.p[ite]=(simu.w[ite]+par.g)/simu.Y[ite]
        

        simu.gamma2[ite]=(np.array(par.sigma + (par.t*par.tests/(1 + simu.I[ite]*par.rho)**par.mu)))

        simu.gamma3[ite]=(np.array(par.gamma1 * (1+ par.kappa1/(1+simu.Q[ite]**(1/par.kappa2)))))

        simu.pi[ite]=(simu.Y[ite]*simu.p[ite] -(simu.lw[ite]+simu.Q[ite])*par.W - simu.lo[ite]*par.G*par.W - (par.xi1*simu.I[ite])**2 - par.xi2*par.d*simu.R[ite])

        simu.util[ite]=(par.varsigma*simu.l[ite]**par.alpha*simu.Y[ite]**(1-par.alpha)+simu.w[ite]+par.g-simu.p[ite]*simu.Y[ite]-par.Z*par.phi2*simu.I[ite]*simu.l[ite]*(1-simu.R[ite]-simu.Q[ite])- par.Z*par.phi3*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite]))

        simu.c[ite]=simu.w[ite]+par.g-simu.p[ite]*simu.Y[ite]

        simu.Pos[ite]=simu.gamma2[ite]*simu.I[ite]/(par.tests)*100

        simu.I[ite+1]=(max(min((1-par.gamma1-simu.gamma2[ite])*simu.I[ite] + par.phi1*simu.s[ite]*simu.wi[ite] + par.phi2*simu.S[ite]*simu.I[ite]*simu.l[ite]*simu.l[ite] + par.phi3*simu.S[ite]*simu.I[ite],1),1.0e-9))

        simu.Q[ite+1]=(max(min((1- simu.gamma3[ite])*simu.Q[ite] + simu.gamma2[ite]*simu.I[ite],1),1.0e-9))

        simu.R[ite+1]=(max(min(simu.R[ite] + par.gamma1*simu.I[ite] + simu.gamma3[ite]*simu.Q[ite],1),1.0e-9))
        ite+=1
    simu.grid = np.linspace(0,ite,ite)
    simu.I = simu.I[0:ite]
    simu.Q = simu.Q[0:ite]
    simu.R = simu.R[0:ite]
    simu.GDP = simu.p*simu.Y
    return(simu)

def simu_bau(par, sol):
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
    simu.c=np.zeros([par.simN])
    simu.I=np.zeros([par.simN+1])
    simu.Q=np.zeros([par.simN+1])
    simu.R=np.zeros([par.simN+1])
    simu.w=np.zeros([par.simN])
    simu.Pos=np.zeros([par.simN])
    simu.I[0]=(par.I_ini)
    simu.Q[0]=(par.Q_ini)
    simu.R[0]=(par.R_ini)
    simu.lw =np.zeros([par.simN])
    simu.lw[0] = 1
    ite=0
    while ite < par.simN:
    #Start of simulation.

        simu.lw[ite]= 1-simu.Q[ite]-simu.R[ite]*par.d

        simu.S[ite]=(1-simu.I[ite]-simu.Q[ite]-simu.R[ite])

        simu.lo[ite]=(1 - simu.lw[ite] - simu.Q[ite] - par.D*simu.R[ite])

        simu.s[ite]=(max((simu.lw[ite]-(1-par.D)*simu.R[ite])*(1-simu.I[ite]/(simu.S[ite]+simu.I[ite])),1.0e-8))

        simu.wi[ite]=(max((simu.lw[ite]-(1-par.D)*simu.R[ite])*(simu.I[ite]/(simu.S[ite]+simu.I[ite])),1.0e-8))

        simu.Y[ite]=(max(par.H*np.log(par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])-(par.chi*simu.I[ite])**2 - par.varphi*simu.R[ite], 1.0e-4))
        
        simu.w[ite]=(simu.lo[ite]*par.G+simu.lw[ite]+simu.Q[ite])*par.W

        simu.l[ite] = 1

        if  simu.l[ite] < 0:
            simu.l[ite] = 1.0e-8
            simu.p[ite] = 1.0e-8

        elif simu.l[ite] >= 1:
            simu.l[ite]=1
            simu.p[ite]=((1-par.alpha)*par.varsigma)/(simu.Y[ite]**par.alpha)
        else:
            simu.p[ite]=(1-par.alpha)*par.varsigma*simu.l[ite]**(par.alpha) * simu.Y[ite]**(-par.alpha)

        if simu.p[ite]*simu.Y[ite]>simu.w[ite]+par.g:
            simu.p[ite]=(simu.w[ite]+par.g)/simu.Y[ite]
        
        #if simu.Y[ite]<=1.0e-7:
            #simu.p[ite]=1.0e-8
        

        simu.gamma2[ite]=(np.array(par.sigma + par.t*par.tests/(1 + simu.I[ite]*par.rho)**par.mu))

        simu.gamma3[ite]=(np.array(par.gamma1 * (1+ par.kappa1/(1+simu.Q[ite]**(1/par.kappa2)))))

        simu.pi[ite]=(simu.Y[ite]*simu.p[ite] -(simu.lw[ite]+simu.Q[ite])*par.W - simu.lo[ite]*par.G*par.W - (par.xi1*simu.I[ite])**2 - par.xi2*par.d*simu.R[ite])

        simu.util[ite]=(par.varsigma*simu.l[ite]**par.alpha*simu.Y[ite]**(1-par.alpha)+simu.w[ite]+par.g-simu.p[ite]*simu.Y[ite]-par.Z*par.phi2*simu.I[ite]*simu.l[ite]*(1-simu.R[ite]-simu.Q[ite])- par.Z*par.phi3*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite]))

        simu.c[ite]=simu.w[ite]+par.g-simu.p[ite]*simu.Y[ite]

        simu.Pos[ite]=simu.gamma2[ite]*simu.I[ite]/(par.tests)*100

        simu.I[ite+1]=(max(min((1-par.gamma1-simu.gamma2[ite])*simu.I[ite] + par.phi1*simu.s[ite]*simu.wi[ite] + par.phi2*simu.S[ite]*simu.I[ite]*simu.l[ite]*simu.l[ite] + par.phi3*simu.S[ite]*simu.I[ite],1),1.0e-9))

        simu.Q[ite+1]=(max(min((1- simu.gamma3[ite])*simu.Q[ite] + simu.gamma2[ite]*simu.I[ite],1),1.0e-9))

        simu.R[ite+1]=(max(min(simu.R[ite] + par.gamma1*simu.I[ite] + simu.gamma3[ite]*simu.Q[ite],1),1.0e-9))
        ite+=1
    simu.grid = np.linspace(0,ite,ite)
    simu.I = simu.I[0:ite]
    simu.Q = simu.Q[0:ite]
    simu.R = simu.R[0:ite]
    simu.GDP = simu.p*simu.Y
    return(simu)

def simu_of(par, sol):
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
    simu.c=np.zeros([par.simN])
    simu.I=np.zeros([par.simN+1])
    simu.Q=np.zeros([par.simN+1])
    simu.R=np.zeros([par.simN+1])
    simu.w=np.zeros([par.simN])
    simu.Pos=np.zeros([par.simN])
    simu.I[0]=(par.I_ini)
    simu.Q[0]=(par.Q_ini)
    simu.R[0]=(par.R_ini)
    simu.lw =np.zeros([par.simN])
    simu.lw[0] = 1
    ite=0
    while ite < par.simN:
    #Start of simulation.
        simu.lw[ite]= 1-simu.Q[ite]-simu.R[ite]*par.d

        if ite == 0:

            simu.lw[ite]=1

        simu.S[ite]=(1-simu.I[ite]-simu.Q[ite]-simu.R[ite])

        simu.lo[ite]=(1 - simu.lw[ite] - simu.Q[ite] - par.D*simu.R[ite])

        simu.s[ite]=(max((simu.lw[ite]-(1-par.D)*simu.R[ite])*(1-simu.I[ite]/(simu.S[ite]+simu.I[ite])),1.0e-9))

        simu.wi[ite]=(max((simu.lw[ite]-(1-par.D)*simu.R[ite])*(simu.I[ite]/(simu.S[ite]+simu.I[ite])),1.0e-9))

        simu.Y[ite]=(max(par.H*np.log(par.upsillon*simu.lw[ite]+par.Upsillon*simu.lo[ite])-(par.chi*simu.I[ite])**2 - par.varphi*simu.R[ite], 1.0e-9))

        simu.l[ite]=(par.Z*par.phi2*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite])/(par.alpha*par.varsigma))**(1/(par.alpha-1))*simu.Y[ite]
        
        simu.w[ite]=(simu.lo[ite]*par.G+simu.lw[ite]+simu.Q[ite])*par.W

        if  simu.l[ite] < 0:
            simu.l[ite]=0
            simu.p[ite] = 0

        elif simu.l[ite]>1:
            simu.l[ite]=1
            simu.p[ite]=((1-par.alpha)*par.varsigma)/(simu.Y[ite]**par.alpha)
        else:
            simu.p[ite]=(1-par.alpha)*par.varsigma*simu.l[ite]**(par.alpha) * simu.Y[ite]**(-par.alpha)

        if simu.p[ite]*simu.Y[ite]>simu.w[ite]+par.g:
            simu.p[ite]=(simu.w[ite]+par.g)/simu.Y[ite]
        

        simu.gamma2[ite]=(np.array(par.sigma + par.t*par.tests/(1 + simu.I[ite]*par.rho)**par.mu))

        simu.gamma3[ite]=(np.array(par.gamma1 * (1+ par.kappa1/(1+simu.Q[ite]**(1/par.kappa2)))))

        simu.pi[ite]=(simu.Y[ite]*simu.p[ite] -(simu.lw[ite]+simu.Q[ite])*par.W - simu.lo[ite]*par.G*par.W - (par.xi1*simu.I[ite])**2 - par.xi2*par.d*simu.R[ite])

        simu.util[ite]=(par.varsigma*simu.l[ite]**par.alpha*simu.Y[ite]**(1-par.alpha)+simu.w[ite]+par.g-simu.p[ite]*simu.Y[ite]-par.Z*par.phi2*simu.I[ite]*simu.l[ite]*(1-simu.R[ite]-simu.Q[ite])- par.Z*par.phi3*simu.I[ite]*(1-simu.R[ite]-simu.Q[ite]))

        simu.c[ite]=simu.w[ite]+par.g-simu.p[ite]*simu.Y[ite]

        simu.Pos[ite]=simu.gamma2[ite]*simu.I[ite]/(par.tests)*100

        simu.I[ite+1]=(max(min((1-par.gamma1-simu.gamma2[ite])*simu.I[ite] + par.phi1*simu.s[ite]*simu.wi[ite] + par.phi2*simu.S[ite]*simu.I[ite]*simu.l[ite]*simu.l[ite] + par.phi3*simu.S[ite]*simu.I[ite],1),1.0e-9))

        simu.Q[ite+1]=(max(min((1- simu.gamma3[ite])*simu.Q[ite] + simu.gamma2[ite]*simu.I[ite],1),1.0e-9))

        simu.R[ite+1]=(max(min(simu.R[ite] + par.gamma1*simu.I[ite] + simu.gamma3[ite]*simu.Q[ite],1),1.0e-9))


        ite+=1
    simu.grid = np.linspace(0,ite,ite)
    simu.I = simu.I[0:ite]
    simu.Q = simu.Q[0:ite]
    simu.R = simu.R[0:ite]
    simu.GDP = simu.p*simu.Y
    return(simu)