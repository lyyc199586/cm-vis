# calculations in Nuc-PFM
import numpy as np


class LDL:
    def __init__(self, props: list[float]):
        """props: [sigma_ts, sigma_cs, mu, K, Gc, ell, h]"""
        self.props = props

    def delta(self) -> float:
        """use h-correction if h is not 0"""
        sts, scs, mu, k, gc, ell, h = self.props
        E = 9 * mu * k / (mu + 3 * k)
        shs = 2 / 3 * sts * scs / (scs - sts)
        W_ts = sts**2 / 2 / E
        if(h != 0):
            delta = (
                pow(1 + 3.0 / 8.0 * h / ell, -2) * (sts + 3 * (1 + 2 * np.sqrt(3)) * shs)
                / (8 + 3 * np.sqrt(3)) / shs * 3 / 16 * (gc / W_ts / ell)
                + pow(1 + 3.0 / 8.0 * h / ell, -1) * 2 / 5
            )
        else:
            delta = (sts + (1 + 2 * np.sqrt(3)) * shs) / (8 + 3 * np.sqrt(3)) / shs * 3 / 16 * gc / W_ts / ell + 3 / 8
        
        return delta

    def l_tip(self) -> float:
        """calc l_tip = l*sqrt(delta + 1)"""
        _, _, _, _, _, ell, _ = self.props
        delta = self.delta()

        return ell * np.sqrt(delta + 1)

    def ce(self, stress: list[float]) -> float:
        """stress = [s1, s2, s3]"""
        # unpack data
        s1, s2, s3 = stress
        sts, scs, mu, k, gc, ell, _ = self.props
        # calc properties
        E = 9 * mu * k / (mu + 3 * k)
        lbda = k - 2 / 3 * mu
        shs = 2 / 3 * sts * scs / (scs - sts)
        W_ts = sts**2 / 2 / E
        W_hs = shs**2 / 2 / k
        # calc invariants
        I1 = s1 + s2 + s3
        J2 = 1 / 6 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)
        delta = self.delta()
        a1 = -1 / shs * delta * gc / 8 / ell + 2 / 3 * W_hs / shs
        a2 = -(
            np.sqrt(3) * (3 * shs - sts) / shs / sts * delta * gc / 8 / ell
            + 2 / np.sqrt(3) * W_hs / shs
            - 2 * np.sqrt(3) * W_ts / sts
        )
        ce = (
            a2 * np.sqrt(J2)
            + a1 * I1
            + (1 - np.sign(I1)) * (J2 / 2 / mu + I1 * I1 / (6 * (3 * lbda + 2 * mu)))
        )

        return ce

    def f(self, stress: list[float]) -> float:
        # unpack data
        s1, s2, s3 = stress
        sts, scs, mu, k, gc, ell, _ = self.props
        # calc invariants
        I1 = s1 + s2 + s3
        J2 = 1 / 6 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)
        # get delta and ce
        delta = self.delta()
        ce = self.ce()
        f = J2 / mu + I1 * I1 / (9 * k) - ce - 3 / 8 * gc * delta / ell

        return f
