# generate strength surface data of various types
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

class StrengthSurface:
    """generate strength surface data of certain types"""

    def __init__(self, mname:str, type:str, props:list, srange:list[float]) -> None:
        """
        mname: material name
        type: strength surface type
        props: material properties needed
        srange: [xmin, xmax, num]
        type                props
        'VMS'               [sigma_y]
        'DRUCKER'           [sigma_ts, sigma_cs]
        'ISOTROPIC'         [sigma_ts, mu, K]
        'VOLDEV'            [sigma_ts, mu, K]
        'SPECTRAL'          [sigma_ts, mu, K]
        'KLBFNUC'           [sigma_ts, sigma_cs, mu, K, Gc, ell, delta]
        'KLRNUC'            [sigma_ts, sigma_cs, mu, K, Gc, ell, delta]
        'LDLNUC'            [sigma_ts, sigma_cs, mu, K, Gc, ell, h]
        """
        self.mname = mname
        self.type = type
        self.props = props
        self.range = srange

    def gen(self, data_dir=None):
        """ if data_dir is not None, store data.npy to data_dir as:
                npy_data: './data_dir/ss_{mname}_{type}_props{props}_srange{srange}.npy' """
        def vms(s1, s2, s3, sigma_y):
            """calculate Von Mises yield surface
            props : [sigma_y]"""
            sigma_v = np.sqrt(0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2))
            f = sigma_v - sigma_y

            return f

        def drucker(s1, s2, s3, sts, scs):
            """calculate Drucker-Prager strength surface:
            f = alpha*I1 + sqrt(J2) - k
            props : [sigma_ts, sigma_cs]"""
            I1 = s1 + s2 + s3
            J2 = 1 / 6 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)
            f = (
                np.sqrt(J2)
                + (scs - sts) * I1 / (np.sqrt(3) * (scs + sts))
                - 2 * scs * sts / (np.sqrt(3) * (scs + sts))
            )

            return f

        def isotropic(s1, s2, s3, sts, mu, k):
            """Strength surface of isotropic phase field model from Kumar et al. (2020):
            f = J2/mu + I1^2/9k - sigma_ts^2/E
            props: [sigma_ts, mu, K]"""
            E = 9 * mu * k / (mu + 3 * k)
            I1 = s1 + s2 + s3
            J2 = 1 / 6 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)
            f = J2 / mu + I1 * I1 / 9 / k - sts * sts / E

            return f

        def voldev(s1, s2, s3, sts, mu, k):
            """Strength surface of phase field model with V-D split
            props: [sigma_ts, mu, K]"""
            E = 9 * mu * k / (mu + 3 * k)
            I1 = (s1 + s2 + s3 + np.abs(s1 + s2 + s3)) / 2
            J2 = 1 / 6 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)
            f = J2 / mu + I1 * I1 / 9 / k - sts * sts / E

            return f
        
        def spectral(s1, s2, s3, sts, mu, k):
            """Strength surface of phase field model with spectral split (d=0)
            from Lorenzis's IJF paper (2021)
            props: [sigma_ts, lbda, mu, K, nu]
            warning: this is very time consuming since I dont know how to do 
            this calculation with ndarrays...
            """
            E = 9 * mu * k / (mu + 3 * k)
            lbda = k - 2/3 * mu
            nu = (3*k - 2*mu) / 2 / (3*k+mu)
            
            def calc(s11, s22, s33, sts, lbda, mu, k, nu) -> float:
                s3, s2, s1 = np.sort(np.array([s11, s22, s33]))
                I1 = s1 + s2 + s3
                J2 = 1 / 6 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)
                if(s3 - nu * (s1 + s2) >= 0):
                    F = I1**2 / 9 / k + J2 / mu -  sts**2 / E
                elif(2 * (lbda + mu) * s2 - lbda * (s1 + s3) >= 0 and s1 + s2 + s3 >= 0):
                    F = (4 * mu**2 * (s1**2 + s2**2) + 2 * lbda * mu *
                         (5 * s1**2 - 2 * s1 * s2 + 5 * s2**2 + s3**2) + lbda**2 *
                         (6 * (s1**2 + s2**2) - 2 * s2 * s3 + 2 * s3**2 - 2 * s1 *
                          (4 * s2 + s3))) / (2 * mu * (3 * lbda + 2 * mu)**2) - sts**2 / E
                elif(2 * (lbda + mu) * s2 - lbda * (s1 + s3) >= 0 and s1 + s2 + s3 <= 0):
                    F = ((2 * (lbda + mu) * s1 - lbda * (s2 + s3))**2 +
                         ((2 * (lbda + mu) * s2 - lbda * (s1 + s3))**2)) / (2 * mu * (3 * lbda + 2 * mu)**2) - sts**2 / E
                elif(2 * mu * s1 + lbda * (2 * s1 - s2 - s3) >= 0 and s1 + s2 + s3 >= 0):
                    F = ((10 * lbda * mu + 4 * mu**2) * s1**2 + lbda *
                         (2 * mu * (s2 + s3)**2 + lbda *
                          (-2 * s1 + s2 + s3)**2)) / (2 * mu * (3 * lbda + 2 * mu)**2) - sts**2 / E
                elif(2 * (lbda + mu) * s1 - lbda * (s2 + s3) >= 0 and s1 + s2 + s3 <= 0):
                    F = (2 * (lbda + mu) * s1 - lbda * (s2 + s3))**2 / (2 * mu * (3 * lbda + 2 * mu)**2) - sts**2 / E
                else:
                    F = -(I1**2 / 9 / k + J2 / mu) -  sts**2 / E
                                                
                return F
            
            f = np.zeros_like(s1)
            for ii in range(np.size(s1, 0)):
                for jj in range(np.size(s2, 0)):
                    for kk in range(np.size(s3, 0)):
                        f[ii, jj, kk] = calc(s1[ii, jj, kk], s2[ii, jj, kk], s3[ii, jj , kk],
                                          sts, lbda, mu, k, nu)
            
            return f

        def klbfnuc(s1, s2, s3, sts, scs, mu, k, gc, ell, delta):
            """calculate the strength surface of the nucleation phase field model introduced
            by Kumar et al. (2020)
            props: [sigma_ts, sigma_cs, mu, K, Gc, ell, delta]"""
            I1 = s1 + s2 + s3
            J2 = 1 / 6 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)
            
            # parameters
            b0 = delta * 3 / 8 * gc / ell
            b1 = (-(1 + delta) * (scs - sts) / (2 * scs * sts) * 3 / 8 * gc / ell -
                   (8 * mu + 24 * k - 27 * sts) * (scs - sts) / (144 * mu * k) -
                   (mu + 3 * k) * (scs**3 - sts**3) * sts /
                   (18 * mu**2 * k**2) * ell / gc)
            b2 = (-np.sqrt(3) * (1 + delta) * (scs + sts) / (2 * scs * sts) * 3 / 8 * gc / ell +
                   (8 * mu + 24 * k - 27 * sts) * (scs + sts) /
                   (48 * np.sqrt(3) * mu * k) + (mu + 3 * k) * (scs**3 + sts**3) * sts /
                   (6 * np.sqrt(3) * mu**2 * k**2) * ell / gc)
            b3 = (sts / mu / k) * ell / gc
            ce = 1 / (1 + b3 * I1**2) * (b2 * np.sqrt(J2) + b1 * I1 + b0)
            f = J2 / mu + I1*I1 / 9 / k - ce - 3 / 8 * gc / ell
            
            return f
        
        def klrnuc(s1, s2, s3, sts, scs, mu, k, gc, ell, delta):
            """calculate the strength surface of the nucleation phase field model introduced
            by Kumar et al. (2022)
            props: [sigma_ts, sigma_cs, mu, lbda, K, Gc, ell, delta]"""
            lbda = k - 2/3 * mu
            I1 = s1 + s2 + s3
            J2 = 1 / 6 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)
            
            b0 = delta * 3 / 8 * gc / ell
            b1 = (- (1 + delta) * (scs - sts) / (2 * scs * sts) * 3 / 8 * gc / ell
                    + sts/(6*(3*lbda + 2*mu)) + sts/6/mu)
            b2 = (- np.sqrt(3) * (1 + delta) * (scs + sts) / (2 * scs * sts) * 3 /8 * gc / ell
                    + sts/(2*np.sqrt(3)*(3*lbda + 2*mu)) + sts/2/np.sqrt(3)/mu)
            ce = b2 * np.sqrt(J2) + b1 * I1 + b0 + (1 - np.sign(I1))*(J2/2/mu + I1*I1/(6*(3*lbda + 2*mu)));
            f = J2 / mu + I1*I1 / (9 * k) - ce - 3 / 8 * gc / ell
            
            return f
        
        def ldlnuc(s1, s2, s3, sts, scs, mu, k, gc, ell, h):
            """calculate the strength surface of the nucleation phase field model introduced
            by Larsen et al. (2024)
            props: [sigma_ts, sigma_cs, mu, K, Gc, ell, h (use h-correction if provided)]"""
            E = 9 * mu * k / (mu + 3 * k)
            lbda = k - 2/3 * mu
            shs = 2/3*sts*scs/(scs - sts)
            
            I1 = s1 + s2 + s3
            J2 = 1 / 6 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)
            
            W_ts = sts**2/2/E 
            W_hs = shs**2/2/k
            
            if(h != 0):
                delta = (
                    pow(1 + 3.0 / 8.0 * h / ell, -2) * (sts + 3 * (1 + np.sqrt(3.0)) * shs)
                    / (3 + 10 * np.sqrt(3.0)) / shs * 3 / 16 * (gc / W_ts / ell)
                    + pow(1 + 3.0 / 8.0 * h / ell, -1) * 2 / 5
                )
            else:
                delta = (sts + 8.15 * shs) / 23.25 / shs * 3 / 16 * gc / W_ts / ell + 3 / 8
                
            a1 = 1 / shs * delta * gc / 8 / ell - 2 / 3 * W_hs / shs
            a2 = (
                np.sqrt(3) * (3 * shs - sts) / shs / sts * delta * gc / 8 / ell
                + 2 / np.sqrt(3) * W_hs / shs
                - 2 * np.sqrt(3) * W_ts / sts
            )
            
            ce = a2 * np.sqrt(J2) + a1 * I1 - (1 - np.sign(I1)) * (J2/2/mu + I1*I1/(6*(3*lbda + 2*mu)))
            
            f = J2 / mu + I1*I1 / (9 * k) + ce - 3 / 8 * gc * delta / ell
            return f

        # generate stress space
        x_ = np.linspace(self.range[0], self.range[1], num=self.range[2])
        y_ = np.linspace(self.range[0], self.range[1], num=self.range[2])
        z_ = np.linspace(self.range[0], self.range[1], num=self.range[2])
        s1, s2, s3 = np.meshgrid(x_, y_, z_, indexing="xy")

        # generate strength surface data
        match self.type:
            case "VMS":
                sigma_y = self.props[0]
                f = vms(s1, s2, s3, sigma_y)
            case "DRUCKER":
                sigma_ts, sigma_cs = self.props
                f = drucker(s1, s2, s3, sigma_ts, sigma_cs)
            case "ISOTROPIC":
                sigma_ts, mu, k = self.props
                f = isotropic(s1, s2, s3, sigma_ts, mu, k)
            case "VOLDEV":
                sigma_ts, mu, k = self.props
                f = voldev(s1, s2, s3, sigma_ts, mu, k)
            case "SPECTRAL":
                sigma_ts, lbda, mu, k, nu = self.props
                f = spectral(s1, s2, s3, sigma_ts, lbda, mu, k, nu)
            case "KLBFNUC":
                sigma_ts, sigma_cs, mu, k, gc, ell, delta = self.props
                f = klbfnuc(s1, s2, s3, sigma_ts, sigma_cs, mu, k, gc, ell, delta)
            case "KLRNUC":
                sigma_ts, sigma_cs, mu, k, gc, ell, delta = self.props
                f = klrnuc(s1, s2, s3, sigma_ts, sigma_cs, mu, k, gc, ell, delta)
            case "LDLNUC":
                sigma_ts, sigma_cs, mu, k, gc, ell, h = self.props
                f = ldlnuc(s1, s2, s3, sigma_ts, sigma_cs, mu, k, gc, ell, h)
        
        # plot 2D contours for illustration
        fig, ax = plt.subplots()
        ax.set(aspect="equal", xlabel="s1", ylabel="s2", title=f"{self.mname}_{self.type}")
        xmin, xmax, num = self.range
        dx = (xmax - xmin) / (num - 1)
        z_index = int((0 - xmin) / dx)
        contours = find_contours(f[:, :, z_index], 0)
        for contour in contours:
            ax.plot(contour[:, 0] * dx + xmin, contour[:, 1] * dx + xmin)
        
        # save data as data_dir.npy
        data_file = f'/ss_{self.mname}_{self.type}_props{self.props}_srange{self.range}.npy'
        if (data_dir is not None):
            np.save(data_dir + data_file, f)
            print(f"Saved to: {data_dir + data_file}")
