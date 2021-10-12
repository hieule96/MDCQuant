# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 09:43:38 2021

@author: hieu1
"""
import Transform as tf
import numpy as np
import logging
from scipy.optimize import curve_fit
import scipy.optimize
import scipy.stats
import Quantlib as quant
import Quadtreelib as qd
import pdb
import matplotlib.pyplot as plt

NBCOEFFR = 4
NBCOEFFD = 4
NBCOEFFDR = 3

def init_logger(*, fn=None):

    # !!! here
    from imp import reload # python 2.x don't need to import reload, use it directly
    reload(logging)

    logging_params = {
        'level': logging.INFO,
        'format': '%(message)s',
    }

    if fn is not None:
        logging_params['filename'] = fn
    logging.basicConfig(**logging_params)
    # logging.getLogger().addHandler(logging.StreamHandler())
    logging.error('init basic configure of logging success')
class OptimizerParamterForComputeLocale():
    def __init__(self, lam1_min=np.nan, lam2_min=np.nan, mu1=np.nan, mu2=np.nan, Ci1=np.nan, Ci2=np.nan, sigma=np.nan, lam1 = np.nan,lam2=np.nan,lam1_max=np.nan, lam2_max=np.nan, E1=np.nan, E2=np.nan):
        self.Ci1 = Ci1
        self.Ci2 = Ci2
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam1_min = lam1_min
        self.lam2_min = lam2_min
        self.lam1_max = lam1_max
        self.lam2_max = lam2_max
        self.mu1 = mu1
        self.mu2 = mu2
        self.E1 = E1
        self.E2 = E2
        self.sigma = sigma
    def DebugStructure(self):
        print (
        self.Ci1,
        self.Ci2,
        self.lam1,
        self.lam2,
        self.lam1_min,
        self.lam2_min,
        self.lam1_max,
        self.lam2_max,
        self.mu1,
        self.mu2,
        self.E1,
        self.E2,
        self.sigma)
class OptimizerParameterCBR(OptimizerParamterForComputeLocale):
    def __init__(self, lam1_min, lam2_min, mu1, mu2,
             n0, LCU, Rt, Dm,
             QPmax=51, iteration_limit = 1,logname="Optimizer_log.txt",rN=0.5):
        super().__init__(lam1_min, lam2_min, mu1, mu2)
        self.n0 = n0
        self.LCU = LCU
        self.Rt = Rt
        self.Dm = Dm
        self.QPmax = QPmax
        self.logname = logname
        self.rN = rN
    def getlam1min(self):
        return self.lam1_min
    def getlam2min(self):
        return self.lam2_min
    def getmu1(self):
        return self.mu1
    def getmu2(self):
        return self.mu2
    def set_Rt(self,Rt):
        self.Rt = Rt
class OptimizerParameterLambdaCst(OptimizerParamterForComputeLocale):
    def __init__(self,lam1,lam2, mu1, mu2,
         n0, LCU, Dm,
         QPmax=51,logname="Optimizer_log.txt",rN=0.1):
        super().__init__(lam1=lam1,lam2=lam2,mu1=mu1,mu2=mu2)
        self.n0 = n0
        self.LCU = LCU
        self.Dm = Dm
        self.logname = logname
        self.rN = rN

class ComputeStatistic():
    @staticmethod
    def computeSigmaNodeAC(node,LCU):
        img_cu = node.get_points(LCU.img)
        img_dct_cu_AC = tf.dct(img_cu - img_cu.mean())
        sigma = np.std(img_dct_cu_AC)
        return sigma
    @staticmethod
    def get_mse(img1, img2):
        diff = np.subtract(img1[:], img2[:])
        MSE = np.square(diff).mean()
        return MSE
    @staticmethod
    def entropy_node(img_dct_cu):
        # Calculate entropy
        entropy = 0
        _,counts =np.unique(img_dct_cu,return_counts=True)
        probalitity = counts/counts.sum()
        # n_classes = np.count_nonzero(probalitity)
        entropy = -np.sum(probalitity*np.log2(probalitity))
        return entropy
    @staticmethod
    def get_rsquare(data_fit,real_data):
        residuals = real_data - data_fit
        ss_tot = np.sum((real_data - np.mean(real_data))**2)
        ss_res = np.sum(residuals**2)
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared
class Rounding():
    @staticmethod
    def roundTo3Decimal(value):
        return np.around(value, decimals=3)
class Optimizer():
    def __init__(self,oparam,StdComputeMethod = ComputeStatistic.computeSigmaNodeAC):
        self.globalparam = oparam
        self.StdComputeMethod = StdComputeMethod
    def get_LCU(self):
        return self.globalparam.LCU
    def getTiles(self):
        return self.globalparam.LCU.tiles
    def dij(self,cu,sigma,Q):
        raise NotImplementedError("Method calculate distortion by cu dij_CU is not implemented")
    def dj(self, qtree, sigma, Q):
        raise NotImplementedError("Method calculate distortion by ctu dj_CTU is not implemented")
    def rij(self,cu,Q):
        raise NotImplementedError("Method rij is not implemented")
    def rj(self,qtree,Q):
        raise NotImplementedError("Method rj is not implemented")
    def getlam1min(self):
        return self.globalparam.lam1_min
    def setRt(self,Rt):
        self.globalparam.Rt = Rt
    def getlam2min(self):
        return self.globalparam.lam2_min
    def getmu1(self):
        return self.globalparam.mu1
    def getmu2(self):
        return self.globalparam.mu2
    def initializeCijSigma(self, qtree, compute_sigma_function,rN):
        """
        Parameters
        ----------
        qtree : Quadtree
        Quadtree
        compute_sigma_function: function to compute sigma (with AC Default quantized or not)
        Returns
        -------
        sigma : array of float
            standard deviation of the imqge
        Ci1 : array of float
            Redundancy parameter of Description 1Ci2 : array of float
            Redundancy parameter of Description 2
        """
        c = qtree
        Ci1 = np.ones(len(c))
        Ci2 = np.ones(len(c))
        sigma = np.zeros(len(c))
        index = range(0, len(c))
        LCU = self.globalparam.LCU
        AllocationSigma = np.zeros(len(c))
        for i in index:
            sigma[i] = compute_sigma_function(c[i],LCU)
            AllocationSigma[i] = sigma[i]
            turn = 0
        while (turn < len(c)):
            max_sigma_index = np.argmax(AllocationSigma)
            if (turn%2==0):
                Ci1[max_sigma_index] = 1
                Ci2[max_sigma_index] = rN
            else:
                Ci1[max_sigma_index] = rN
                Ci2[max_sigma_index] = 1
            AllocationSigma[max_sigma_index] = 0
            turn = turn + 1
        # logging.debug("Sigma: %s" %(sigma))
        # logging.debug("Ci1 %s Ci2 %s " %(Ci1,Ci2))
        return sigma, Ci1, Ci2
    def optimizeQuadtree(self, qtree, compute_sigma_function):
        raise NotImplementedError("Method quatree is not implemented")
    def optimizeQuadtreeLambaCst(self,qtree,compute_sigma_function,lam1,lam2):
        raise NotImplementedError("Method optimizeQuadtreeLambaCst is not implemented")
    @staticmethod
    def deltadr_right(sigmaij, lamj, muj, Ej, Cij, rN):
        """
        Calculate the right side of Equation allocation

        -----------
        Parameters:
        sigmaij : float or numpy_array
            Standard Deviation of a CU
        lamj: float
            Contraint on rate of Description j
        muj: float
            Penality coeficients on distortion of Description j
        Ej: float
            Penality on Distortion of Description j
        Cij: float [0,0.5] or numpy_array
            Decide which CU is coarsed (0.5) coded or fine(1) coded of Description j
        return
        ------------
        value : Right side of equation
        """
        value = -lamj / ((sigmaij ** 2) * ((Cij / (1 + rN)) + muj * Ej))
        return value
    def optimizeTiles(self):
        Q1 = []
        Q2 = []
        D1_image = 0
        D2_image = 0
        R1_image = 0
        R2_image = 0
        self.globalparam.LCU.Qi1 = []
        self.globalparam.LCU.Qi2 = []
        # logname = self.globalparam.logname
        # init_logger(fn=logname)
        # logging.info("Start Optimizer")
        tiles = self.getTiles()
        if (len(tiles)==0):
            raise Exception("No tiles to process ! len(tiles) == 0")
        i=0
        for tile in tiles:
            # logging.info("[INFO] Tile index %s" % (i))
            (Qi1, Qi2, D1, D2, Ratej_Q1, Ratej_Q2) = self.optimizeQuadtreeLambaCst(tile,
                                                                            self.StdComputeMethod)
            # logging.info("D1 %s D2 %s Rate1 %s Rate2 %s"
            #              % (Rounding.roundTo3Decimal(D1),
            #                 Rounding.roundTo3Decimal(D2),
            #                 Rounding.roundTo3Decimal(Ratej_Q1),
            #                 Rounding.roundTo3Decimal(Ratej_Q2)))
            D1_image += D1
            D2_image += D2
            R1_image += Ratej_Q1
            R2_image += Ratej_Q2
            self.globalparam.LCU.Qi1.append(Qi1)
            self.globalparam.LCU.Qi2.append(Qi2)
            Q1.append(Qi1)
            Q2.append(Qi2)
            i = i+1
        nbTiles = len(tiles)
        # logging.info("D1_image %s ,D2_image %s ,R1_image %s ,R2_image %s "
        # % (D1_image / nbTiles,
        # D2_image / nbTiles,
        # R1_image / nbTiles,
        # R2_image / nbTiles))
        return Q1, Q2, D1_image / nbTiles, D2_image / nbTiles, R1_image / nbTiles, R2_image / nbTiles
    def optimize_LCU(self):
        Q1 = []
        Q2 = []
        D1_image = 0
        D2_image = 0
        R1_image = 0
        R2_image = 0
        self.globalparam.LCU.Qi1 = []
        self.globalparam.LCU.Qi2 = []
        LCU = self.get_LCU()
        for i in range(LCU.nbCTU):
            (Qi1, Qi2, D1, D2, Ratej_Q1, Ratej_Q2) = self.optimizeQuadtreeLambaCst(LCU.CTUs[i],self.StdComputeMethod,self.globalparam.lam1,self.globalparam.lam2)
            D1_image += D1
            D2_image += D2
            R1_image += Ratej_Q1
            R2_image += Ratej_Q2
            self.globalparam.LCU.Qi1.append(Qi1)
            self.globalparam.LCU.Qi2.append(Qi2)
            Q1.append(Qi1)
            Q2.append(Qi2)
        nbCTU = self.globalparam.LCU.nbCTU
        return Q1, Q2, D1_image / nbCTU, D2_image / nbCTU, R1_image / nbCTU, R2_image / nbCTU
class FunctionCurveFitting():
    @staticmethod
    def negExponential(x, a, b):
        return a * np.exp(b * x)
    @staticmethod
    def exponential(x, a, b, c):
        return a * np.exp(b * x) + c
    @staticmethod
    def cubic(x, a, b, c, d):
        return a * x ** 3 + b * x ** 2 + c * x + d
    @staticmethod
    def linear(x, a, b):
        return a * x + b
    @staticmethod
    def derivateExp(x, a, b):
        return -a * b * np.exp(-b * x)
    @staticmethod
    def neg_exponential_function2(x,a,b,c,d):
        return a*np.exp(b*x)+c*np.exp(d*x)
class CoefficientCurveFitting():
    def __init__(self):
        self.DRCoefficients = []
        self.RCoefficients = []
        self.DCoefficients = []
    def appendRCoeffbyCTU(self,Rcoeff):
        self.RCoefficients.append(Rcoeff)
    def appendDCoeffbyCTU(self,Dcoeff):
        self.DCoefficients.append(Dcoeff)
    def appendDRCoeffbyCTU(self,DRcoeff):
        self.DRCoefficients.append(DRcoeff)
class ConvexHull:
    def __init__(self,key,value):
        self.key_uni = {}
        for k,v in zip(key,value):
            if (k in self.key_uni):
                self.key_uni[k]= min(self.key_uni[k],v)
            else:
                self.key_uni[k]=v
        lists = sorted(self.key_uni.items())
        self.key, self.value = zip(*lists)
    def getunique(self):
        return np.array(self.key),np.array(self.value)
class Optimizer_curvefitting(Optimizer):
    def __init__(self,oparam):
        super().__init__(oparam)
    def rij(self,cu,Q):
        return FunctionCurveFitting.cubic(Q,*cu.Rcoeff)
    def rj(self,qtree,Q):
        rij = 0
        i = 0
        for cu in qtree:
            rij = rij + cu.aki*self.rij(cu,Q[i])
            i = i+1
        return rij
    def dijrij(self,cu,Q,sigma):
        rij = 0
        dij = 0
        if (len(cu.DRcoeff)==3 and len(cu.Rcoeff)==4):
            #modeling Rate by cubic function
            rij = self.rij(cu,Q)
            #modeling Distortion by Neg Exp function
            dij = (sigma**2)*FunctionCurveFitting.negExponential(rij,*cu.DRcoeff)
            if (dij<0):
                dij = 0
        return dij,rij
    def djrj(self,qtree,sigma,Q):
        rj = 0
        dj = 0
        i = 0
        for cu in qtree:
            dij,rij = self.dijrij(cu,Q[i],sigma[i])
            rj = rj+rij*cu.aki
            dj = dj+dij*cu.aki
            i = i+1
        return dj,rj
    def computeCurveCoefficient(LCU):
        for cus in LCU.CTUs:
            for cu in cus:
                # print (cu.x0,cu.y0)
                old_coeffDR = None
                Rcoeff,DRcoeff = Optimizer_curvefitting.curve_fitting(LCU,cu,old_coeffDR)
                cu.set_Rcoeff(Rcoeff)
                cu.set_DRcoeff(DRcoeff)
                old_coeffDR = DRcoeff
        return True
    def initCoefficient(LCU):
        if Optimizer_curvefitting.computeCurveCoefficient(LCU)==True:
            print ("Compute curve coefficients sucess")
        else:
            print ("Curve fitting fail, coefficients fail")
        return
    @staticmethod
    #modify to mse compute in reconstruct domain
    def compute_mse_entropy_QP(LCU, node, QPs):
        # Convert QP to Q
        Q_values = 2 ** ((1 / 6) * (QPs - 4))
        # Take image in partition and dct
        img_cu = node.get_points(LCU.img)
        # if (img_cu.shape[0]>16):
        #     print ("img_cu size to %dx%d" %(img_cu.shape[0],img_cu.shape[1]))
        img_cu_DC = img_cu.mean()
        img_dct_cu_AC = tf.dct(img_cu-img_cu_DC)
        # img_cu = np.int32(img_cu)
        # img_dct_cu = tf.transformNxN(img_cu,img_cu.shape[0])
        # img_dct_cu_DC = img_dct_cu[0][0]
        # Calculate sigma
        sigma = np.std(img_dct_cu_AC)
        # sigma = np.std(img_dct_cu)
        
        entropy_list = []
        mse_list = []
        i = 0
        mQ =quant.matQuant(img_cu.shape[0],img_cu.shape[1])/16
        if (sigma>0):
            for Q in Q_values:                
                # Quantize the image
                # img_dct_cu_Q = quant.QuantCU(img_dct_cu, Q)
                # img_dct_cu_AC_Reconstruct = quant.DeQuantCU(img_dct_cu_Q, Q)
                img_dct_cu_Q = np.round(img_dct_cu_AC / (mQ*Q))
                img_dct_cu_Reconstruct = img_dct_cu_Q * (mQ*Q)
                
                img_rec_2D = tf.idct(img_dct_cu_Reconstruct) + img_cu_DC
                # img_rec_2D = tf.invTransform(img_dct_cu_Q, img_dct_cu_AC_Reconstruct.shape[0])
                img_rec_2D [img_rec_2D <-128] = -128
                img_rec_2D [img_rec_2D >127] = 127
                
                # Calculate entropy
                entropy = ComputeStatistic.entropy_node(img_dct_cu_Reconstruct)
                #print (img_dct_cu_AC_Q)
                if (entropy==0):
                    # pdb.set_trace()
                    break
                entropy_list.append (entropy)
                # Calculate MSE
                mse = ComputeStatistic.get_mse(img_rec_2D , img_cu)
                mse_list.append(mse)
     
                i = i + 1
        entropy_list = np.array(entropy_list)
        mse_list = np.array(mse_list)
        return mse_list,entropy_list,i,sigma
    @staticmethod    
    def curve_fitting(LCU,node,old_pred=None):
        #Select points to plotting
        QPs = np.arange(1,51,1)
        mseQP,entropyQP,index_max,sigma = Optimizer_curvefitting.compute_mse_entropy_QP(LCU,node,QPs)

        Rcoeff=[]
        DRcoeff = []
        # mse_QP_gpu = cu.asarray(log_mse_QP)
        # entropy_QP_gpu = cu.asarray(entropy_QP)
        if (sigma==0 or np.all(entropyQP==0) or index_max<=4):
            print ("Number of element of list is too small return null node: %s %s" %(node.x0,node.y0))
        #fit model entropy, MSE,
        else:
            try:
                cxvhullEntropyQP = ConvexHull(entropyQP,QPs[:index_max])
                entropy_QP_unique,QPs_unique = cxvhullEntropyQP.getunique()
                log_entropy_QP = np.log(entropy_QP_unique)
                a,b,r,_,_ = scipy.stats.linregress(QPs_unique,log_entropy_QP)
                Rcoeff, pcovR = curve_fit(FunctionCurveFitting.negExponential, QPs_unique, entropy_QP_unique,p0=[np.exp(b),a])
                # R_fit= FunctionCurveFitting.linear(QPs,a,b)
                # err = ComputeStatistic.get_rsquare(R_fit,entropy_QP)
                R_fit= FunctionCurveFitting.negExponential(QPs[:index_max],Rcoeff[0],Rcoeff[1])
                err = ComputeStatistic.get_rsquare(R_fit,entropyQP)
                R_fit= FunctionCurveFitting.negExponential(QPs,Rcoeff[0],Rcoeff[1])
                if (err<0.8):
                    print ("rsquared R(QP): e:%s l:%s" %(err,r))
                    plt.plot(QPs,R_fit)
                    plt.scatter(QPs[:index_max],entropyQP)
                    plt.show()
            except RuntimeError:
                print ("Curve fitting of R at node: %s,%s,%s,%s failed" %(node.x0,node.y0,node.x0+node.width,node.y0+node.height))

            try:
                #First fit the linear function then fit last the function exp
                cvxhullRD = ConvexHull(entropyQP,mseQP)
                entropy_QP_unique,mse_QP_unique  = cvxhullRD.getunique()
                log_mse_QP = np.log(mse_QP_unique)
                a,b,r,_,_ = scipy.stats.linregress(entropy_QP_unique,log_mse_QP)
                DRcoeff=[np.exp(b),a]
                DRcoeff, pcovDR = curve_fit(FunctionCurveFitting.negExponential, entropy_QP_unique, mse_QP_unique,p0=(DRcoeff[0],DRcoeff[1]),ftol=0.05, xtol=0.05)
                DR_fit = FunctionCurveFitting.negExponential(entropy_QP_unique, *DRcoeff)
                err = ComputeStatistic.get_rsquare(DR_fit,mse_QP_unique)
                if (err<0.8):
                    print ("rsquared D(R): e:%s l:%s" %(err,r))
                    plt.scatter(entropyQP, mseQP)
                    plt.plot(entropy_QP_unique,DR_fit)
                    plt.show()
            except RuntimeError:
                print ("Curve fitting of DR at node: %s,%s,%s,%s failed" %(node.x0,node.y0,node.x0+node.width,node.y0+node.height))
                print (entropy_QP_unique,mse_QP_unique)
        return Rcoeff,DRcoeff
    @staticmethod
    def findSolutionRateExpNeg(a,b,sigmaij, lamj, muj, Ej, Cij, rN):
        entropy = 0
        if (sigmaij>0):
            entropy = np.log(-Optimizer.deltadr_right(sigmaij, lamj, muj, Ej, Cij, rN)/(a*b))/(-b)
        if (entropy < 0):
            entropy = 0
        if np.isnan(entropy):
            print (a,b,sigmaij, lamj, muj, Ej, Cij, rN)
        return entropy
    @staticmethod
    def findQPfromRate(x, a, b, entropy):
        return FunctionCurveFitting.linear(x, a, b) - entropy
    @staticmethod
    def lamMaxCompute(derivate_DR_max, sigmaij, muj, Ej, Cij, rN):
        lam_max = -((sigmaij ** 2) * ((Cij / (1 + rN)) + muj * Ej)) * derivate_DR_max
        return lam_max
    @staticmethod
    def findLamMaxQtree(qtree,OpParamLocale,OpParamGlobal):
        cus = qtree
        lam1_end = []
        lam2_end = []
        index = np.arange(0,len(cus)-1)
        for cu_index in index:
            if (len(cus[cu_index].DRcoeff)>=3):
                lam1_end.append(
                    Optimizer_curvefitting.lamMaxCompute(
                        FunctionCurveFitting.derivateExp(0, cus[cu_index].DRcoeff[0], cus[cu_index].DRcoeff[1]),
                        OpParamLocale.sigma[cu_index],
                        OpParamLocale.mu1,
                        OpParamLocale.E1,
                        OpParamLocale.Ci1[cu_index],
                        OpParamGlobal.rN))
                lam2_end.append(
                    Optimizer_curvefitting.lamMaxCompute(
                        FunctionCurveFitting.derivateExp(0, cus[cu_index].DRcoeff[0], cus[cu_index].DRcoeff[1]),
                        OpParamLocale.sigma[cu_index],
                        OpParamLocale.mu2,
                        OpParamLocale.E2,
                        OpParamLocale.Ci2[cu_index],
                        OpParamGlobal.rN))
            else:
                lam1_end.append(0)
                lam2_end.append(0)
        return np.max(lam1_end), np.max(lam2_end)
    # def cost_function(qtree):
    @staticmethod
    def cost_func(r,aki,DR_coeff,Ci,lam,mu,Dm):
        DR_coeff = np.hsplit(DR_coeff,2)
        DR_coeff[0] = DR_coeff[0].reshape(-1,)
        DR_coeff[1] = DR_coeff[1].reshape(-1,)
        #pdb.set_trace()
        Di = aki*DR_coeff[0]*np.exp(DR_coeff[1]*r)
        Dcost = np.sum(Ci*Di)
        Dtotal = np.sum(Di)
        R = np.dot(aki,r)
        cost = Dcost + lam*R + mu*((np.abs(Dtotal-Dm) + (Dtotal-Dm))/2)**2 
        print ("cost %s" %(cost))
        return cost
    @staticmethod
    def grad_func(r,aki,DR_coeff,Ci,lam,mu,Dm):
        DR_coeff = np.hsplit(DR_coeff,2)
        DR_coeff[0] = DR_coeff[0].reshape(-1,)
        DR_coeff[1] = DR_coeff[1].reshape(-1,)
        #pdb.set_trace()
        Di = aki*DR_coeff[0]*np.exp(DR_coeff[1]*r)
        Dtotal = np.sum(Di)
        Diprime = aki*DR_coeff[0]*DR_coeff[1]*np.exp(DR_coeff[1]*r)
        Riprime = aki
        E = 2*(np.abs(Dtotal-Dm) + (Dtotal-Dm))
        grad = Ci*Diprime + lam * Riprime + mu*E*Diprime
        #print ("gradiant %s" %(grad))
        return grad        
    @staticmethod
    def cost_function_QP(QP,aki,DR_coeff,R_Coeff,Ci,lam,mu,Dm):
        DR_coeff = np.hsplit(DR_coeff,2)
        DR_coeff[0] = DR_coeff[0].reshape(-1,)
        DR_coeff[1] = DR_coeff[1].reshape(-1,)
        R_Coeff = np.hsplit(R_Coeff,2)
        R_Coeff[0] = R_Coeff[0].reshape(-1,)
        R_Coeff[1] = R_Coeff[1].reshape(-1,)
        r = R_Coeff[0] * np.exp(R_Coeff[1]*QP)
        Di = aki*DR_coeff[0]*np.exp(DR_coeff[1]*r)
        Dcost = np.sum(Ci*Di)
        Dtotal = np.sum(Di)
        R = np.dot(aki,r)
        cost = Dcost + lam*R + mu*((np.abs(Dtotal-Dm) + (Dtotal-Dm))/2)**2 
        # pdb.set_trace()
        # print ("cost %s" %(cost))
        return cost
    @staticmethod
    def grad_func_QP(QP,aki,DR_coeff,R_Coeff,Ci,lam,mu,Dm):
        DR_coeff = np.hsplit(DR_coeff,2)
        DR_coeff[0] = DR_coeff[0].reshape(-1,)
        DR_coeff[1] = DR_coeff[1].reshape(-1,)
        R_Coeff = np.hsplit(R_Coeff,2)
        R_Coeff[0] = R_Coeff[0].reshape(-1,)
        R_Coeff[1] = R_Coeff[1].reshape(-1,)
        r = R_Coeff[0] * np.exp(R_Coeff[1]*QP)
        #pdb.set_trace()
        Di = aki*DR_coeff[0]*np.exp(DR_coeff[1]*r)
        Dtotal = np.sum(Di)
        # Diprime = a1*a2*b1*b2*e^((a2b1+1)e⁽b2x)
        Diprime = aki*DR_coeff[0]*DR_coeff[1]*R_Coeff[0]*R_Coeff[1]*np.exp((R_Coeff[0]*DR_coeff[1]+1)*np.exp(R_Coeff[1]*QP))
        Riprime = aki*R_Coeff[0]*R_Coeff[1]*np.exp(R_Coeff[1]*QP)
        E = 2*(np.abs(Dtotal-Dm) + (Dtotal-Dm))
        grad = Ci*Diprime + lam * Riprime + mu*E*Diprime
        # print ("gradiant %s" %(grad))
        return grad  
        
    @staticmethod
    def compute_QP(qtree,sigma,lami,mui,Ei,Ci,OpParamGlobal):
        QP = []
        DR_coeff_cu = []
        R_coeff_cu=[]
        aki = []
        r = []
        for cu in qtree:
            if (len(cu.DRcoeff)>0):
                DR_coeff_cu.append(cu.DRcoeff)
                R_coeff_cu.append(cu.Rcoeff)
                r.append(1.0)
            else:
                DR_coeff_cu.append(np.array([0,0]))
                R_coeff_cu.append(np.array([0,0]))
                r.append(0)
            aki.append(cu.aki)
        DR_coeff_cu = np.array(DR_coeff_cu)
        R_coeff_cu = np.array(R_coeff_cu)
        aki = np.array(aki)
        r = np.array(r)
        bounds = np.full((r.size,2),[0,51])
        #Solving to QP directly
        result_minize = scipy.optimize.minimize(Optimizer_curvefitting.cost_function_QP,
                                          r,
                                          jac=Optimizer_curvefitting.grad_func_QP,
                                          args=(aki,DR_coeff_cu,R_coeff_cu,Ci,lami,mui,OpParamGlobal.Dm),
                                          bounds=bounds,method='L-BFGS-B')
        pdb.set_trace()
        # R_coeff_cu = np.hsplit(R_coeff_cu,2)
        # R_coeff_cu[0] = R_coeff_cu[0].reshape(-1,)
        # R_coeff_cu[1] = R_coeff_cu[1].reshape(-1,)
        # R_coeff_cu[0][R_coeff_cu==0] = 1
        # R_coeff_cu[1][R_coeff_cu==0] = 1
        # entropy.x[entropy.x==0]=1
        # QP = (np.log(entropy.x) - np.log(R_coeff_cu[0]))/R_coeff_cu[1]
        # pdb.set_trace()
        # np.nan_to_num(QP,copy=False,nan=0.0)
        QP = np.round(result_minize.x)        
        QP[QP>51] = 51
        QP[QP<0] = 0
        # for cu in qtree:
        #     if (len(cu.DRcoeff)>0):
        #         entropy = Optimizer_curvefitting.findSolutionRateExpNeg(cu.DRcoeff[0],
        #                                           cu.DRcoeff[1],
        #                                           sigma[i],
        #                                           lami,
        #                                           mui,
        #                                           Ei,
        #                                           Ci[i],
        #                                           OpParamGlobal.rN)
        #         QP_sol = fsolve(Optimizer_curvefitting.findQPfromRate, [15.0], args=(*cu.Rcoeff, entropy))

        #         QP_sol = 0 if QP_sol<0 else QP_sol
        #         # if QP is superior to QPMax then QP = QpMax
        #         QP_sol = OpParamGlobal.QPmax if QP_sol>OpParamGlobal.QPmax else QP_sol
        #         QP.append(QP_sol)
        #         assert entropy >= 0, "entropy:%s" % (entropy)
        #         assert QP_sol >= 0, "QP1 %s" % (QP_sol)
        #         i = i + 1
        #     else:
        #         QP.append(35)
        return QP
    # Direct lambda injection mode
    def optimizeQuadtreeLambaCst(self, qtree, compute_sigma_function,lam1,lam2):
        sigma,Ci1_old, Ci2_old = self.initializeCijSigma(qtree,self.StdComputeMethod,self.globalparam.rN)
        QP1 = np.zeros(len(Ci1_old))
        QP2 = np.zeros(len(Ci2_old))
        paramlocal = OptimizerParamterForComputeLocale(self.getlam1min(),
                                                       self.getlam2min(),
                                                       self.getmu1(),
                                                       self.getmu2(),
                                                       Ci1_old,
                                                       Ci2_old,
                                                       sigma,E1=0,E2=0)
        paramlocal.lam1 = lam1
        paramlocal.lam2 = lam2
        QP1 = Optimizer_curvefitting.compute_QP(qtree,paramlocal.sigma,paramlocal.lam1,paramlocal.mu1,paramlocal.E1,paramlocal.Ci1,self.globalparam)
        QP2 = Optimizer_curvefitting.compute_QP(qtree,paramlocal.sigma,paramlocal.lam2,paramlocal.mu2,paramlocal.E2,paramlocal.Ci2,self.globalparam)
        D1,Rj_Q1 = self.djrj(qtree,sigma,QP1)
        D2,Rj_Q2 = self.djrj(qtree,sigma,QP1)
        return QP1,QP2, D1, D2, Rj_Q1, Rj_Q2 
        
    # def optimizeQuadtreeCBR(self, qtree, compute_sigma_function):
        # # compute sigma and Ci1,Ci2 for the first time
        # sigma,Ci1_old, Ci2_old = self.initializeCijSigma(qtree,self.StdComputeMethod,self.globalparam.rN)
        # QP1 = np.zeros(len(Ci1_old))
        # QP2 = np.zeros(len(Ci2_old))
        # paramlocal = OptimizerParamterForComputeLocale(self.getlam1min(),
        #                                                self.getlam2min(),
        #                                                self.getmu1(),
        #                                                self.getmu2(),
        #                                                Ci1_old,
        #                                                Ci2_old,
        #                                                sigma,E1=0,E2=0)
        # Dm_global = self.globalparam.Dm
        # n0_global = self.globalparam.n0
        # iteration_limit_global = self.globalparam.iteration_limit
        # state = 1
        # delta_ditch = self.globalparam.deltaRateTarget
        # for iteration in range (0,iteration_limit_global):
        #     if (state == 1):
        #         paramlocal.lam1_max,paramlocal.lam2_max = Optimizer_curvefitting.findLamMaxQtree(qtree,paramlocal,self.globalparam)
        #         lam1_begin = self.globalparam.lam1_min
        #         lam2_begin = self.globalparam.lam2_min
        #         lam1_end = paramlocal.lam1_max
        #         lam2_end = paramlocal.lam2_max
        #         start=time.time()
        #         while(True):
        #             paramlocal.lam1 = (lam1_begin+lam1_end)/2
        #             QP1 = Optimizer_curvefitting.compute_QP(qtree,paramlocal.sigma,paramlocal.lam1,paramlocal.mu1,paramlocal.E1,paramlocal.Ci1,self.globalparam)
        #             D1,Rj_Q1 = self.djrj(qtree,sigma,QP1)
        #             if (Rj_Q1 - self.globalparam.Rt/2 > 0):
        #                 lam1_begin = paramlocal.lam1
        #             else:
        #                 lam1_end = paramlocal.lam1
        #             Delta1 = lam1_end - lam1_begin
        #             #print (Rj_Q1,D1,paramlocal.lam1,Delta1)
        #             if (Delta1 < delta_ditch):
        #                 logging.info("[BREAK] Step1 finish R1: %s " %(Rounding.roundTo3Decimal(Rj_Q1)))
        #                 logging.info("lam1 : %s/%s " %(Rounding.roundTo3Decimal(paramlocal.lam1),Rounding.roundTo3Decimal(paramlocal.lam1_max)))
        #                 break

        #         while (True):
        #             paramlocal.lam2 = (lam2_begin+lam2_end)/2
        #             QP2 = Optimizer_curvefitting.compute_QP(qtree,paramlocal.sigma,paramlocal.lam2,paramlocal.mu2,paramlocal.E2,paramlocal.Ci2,self.globalparam)
        #             D2,Rj_Q2 = self.djrj(qtree,sigma,QP2)
        #             if (Rj_Q2 - self.globalparam.Rt/2 > 0):
        #                 lam2_begin = paramlocal.lam2
        #                 print ("BIG")
        #             else:
        #                 lam2_end = paramlocal.lam2
        #             Delta2 = lam2_end - lam2_begin
        #             #print (Rj_Q2,D2,paramlocal.lam2,Delta2)
        #             if (Delta2 < delta_ditch):
        #                 logging.info("[BREAK] Step1 finish R2: %s " %(Rounding.roundTo3Decimal(Rj_Q2)))
        #                 logging.info("lam2 : %s/%s " %(Rounding.roundTo3Decimal(paramlocal.lam2),Rounding.roundTo3Decimal(paramlocal.lam2_max)))
        #                 break
        #         end = time.time()
        #         print ("Time %s" %(end-start))
        #     state = 2
        #     if (state == 2):
        #         D1,_ = self.djrj(qtree,sigma,QP1)
        #         D2,_ = self.djrj(qtree,sigma,QP2)
        #         if D1 - Dm_global > 0:
        #             d1 = D1 - Dm_global
        #             paramlocal.mu1 = paramlocal.mu1 + n0_global * d1
        #             paramlocal.E1 = 2 * d1
        #             state = 1
        #         if D2 - Dm_global > 0:
        #             d2 = D2 - Dm_global
        #             paramlocal.mu2 = paramlocal.mu2 + n0_global * d2
        #             paramlocal.E2 = 2 * d2
        #             state = 1
        #         if D1 - Dm_global < 0 and D2 - Dm_global < 0:
        #             logging.info ("[BREAK] Step 2 finish D1: %s D2: %s R1: %s R2: %s"%(Rounding.roundTo3Decimal(D1),
        #                                                                                Rounding.roundTo3Decimal(D2),
        #                                                                                Rounding.roundTo3Decimal(Rj_Q1),
        #                                                                                Rounding.roundTo3Decimal(Rj_Q2)))
        #             break
        #     if (iteration == iteration_limit_global-1):
        #         logging.debug ("[INFO] Optimizer Iteration limit reach %s" %(iteration))
        # return QP1,QP2, D1, D2, Rj_Q1, Rj_Q2