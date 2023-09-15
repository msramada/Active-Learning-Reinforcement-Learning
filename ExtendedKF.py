import numpy as np
import torch

class Model:
    def __init__(self, stateDynamics, measurementDynamics, stateDynamics_prime, measurementDynamics_prime, Q, R):
        self.f = stateDynamics
        self.fprime = stateDynamics_prime
        self.g = measurementDynamics
        self.gprime = measurementDynamics_prime
        self.Q = torch.atleast_2d(Q)
        self.R = torch.atleast_2d(R)  
    
    def TrueTraj(self, x0, u):
        x1 = self.f(x0, u) + torch.sqrt(self.Q)@torch.randn(self.Q.shape[0])
        y1 = self.g(x0) + torch.sqrt(self.R)@torch.randn(self.R.shape[0])
        return x1, y1


class Extended_KF:
    def __init__(self, mean, covariance, Model):
        self.Mean = torch.atleast_2d(mean)
        self.Covariance = torch.atleast_2d(covariance)
        self.Model = Model

    def TimeUpdate(self, u):
        u = torch.atleast_2d(u)
        meanP = self.Model.f(self.Mean, u)
        F = self.Model.fprime(self.Mean, u)
        CovarianceP = F @ self.Covariance @ F.T + self.Model.Q
        return meanP, CovarianceP

    def MeasurementUpdate(self, meanP, CovarianceP, y):
        y = torch.atleast_2d(y)
        gx = self.Model.g(meanP)
        H = self.Model.gprime(meanP)
        L = CovarianceP @ H.T @ torch.inverse(H @ CovarianceP @ H.T + self.Model.R)
        self.Mean = meanP + L @ (y-gx)
        self.Covariance = (torch.eye(CovarianceP.shape[0]) - L @ H) @ CovarianceP
    
    def ApplyEKF(self, u, y):
        u = torch.atleast_2d(u)
        y = torch.atleast_2d(y)
        meanP, CovarianceP = self.TimeUpdate(u)
        self.MeasurementUpdate(meanP, CovarianceP, y)

    def ChangeInitialStates(self, mean_new, cov_new):
        self.Mean = torch.atleast_2d(mean_new)
        self.Covariance = torch.atleast_2d(cov_new)
        