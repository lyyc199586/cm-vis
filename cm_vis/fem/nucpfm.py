"""
Nucleation Phase Field Model Calculations
=========================================

This module provides calculations for nucleation phase field models (Nuc-PFM),
specifically implementing the LDL (Length-Dependent Load) model for crack
nucleation analysis in phase field fracture mechanics.

Classes
-------
LDL : Nucleation phase field model calculations

Notes
-----
Implements the nucleation model formulations for nucleation driving force evaluation
and tip length calculations in phase field fracture mechanics.
"""

# calculations in Nuc-PFM
import numpy as np


class LDL:
    """
    Length-Dependent Load nucleation phase field model.
    
    This class implements calculations for the LDL nucleation model
    in phase field fracture mechanics, including nucleation driving force
    evaluation and characteristic length scale computations.
    
    Parameters
    ----------
    props : list of float
        Material properties as [sigma_ts, sigma_cs, mu, K, Gc, ell, h]
        where:
        - sigma_ts : tensile strength
        - sigma_cs : compressive strength  
        - mu : shear modulus
        - K : bulk modulus
        - Gc : nucleation driving force release rate
        - ell : regularization length
        - h : mesh size parameter
        
    Attributes
    ----------
    props : list of float
        Stored material properties
        
    Examples
    --------
    >>> # Initialize LDL model with material properties
    >>> props = [50.0, 200.0, 80.0e3, 133.3e3, 2.7e-3, 0.015, 0.005]
    >>> ldl = LDL(props)
    >>> 
    >>> # Calculate characteristic parameters
    >>> delta = ldl.delta()
    >>> tip_length = ldl.l_tip()
    >>> 
    >>> # Evaluate nucleation driving force for stress state
    >>> stress = [60.0, 30.0, 0.0]
    >>> critical_energy = ldl.ce(stress)
    """
    
    def __init__(self, props: list[float]):
        """
        Initialize the LDL nucleation model.
        
        Parameters
        ----------
        props : list of float
            Material properties [sigma_ts, sigma_cs, mu, K, Gc, ell, h]
        """
        self.props = props

    def delta(self) -> float:
        """
        Calculate the delta parameter for the LDL model.
        
        Computes the delta parameter which characterizes the nucleation
        behavior. Uses h-correction if mesh size h is non-zero.
        
        Returns
        -------
        float
            Delta parameter for nucleation model
            
        Notes
        -----
        The delta parameter controls the nucleation threshold and is
        calculated differently depending on whether mesh size correction
        is applied (h != 0) or not (h = 0).
        """
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
        """
        Calculate the characteristic tip length.
        
        Computes the characteristic length scale at the crack tip
        as l_tip = ell * sqrt(delta + 1).
        
        Returns
        -------
        float
            Characteristic tip length scale
            
        Notes
        -----
        This length scale characterizes the size of the process zone
        at the crack tip in the nucleation model.
        """
        _, _, _, _, _, ell, _ = self.props
        delta = self.delta()

        return ell * np.sqrt(delta + 1)

    def ce(self, stress: list[float]) -> float:
        """
        Calculate nucleation driving force ce for given stress state.
        
        Evaluates the nucleation driving force ce required for crack nucleation
        under the specified stress state using the LDL model.
        
        Parameters
        ----------
        stress : list of float
            Principal stress components [s1, s2, s3]
            
        Returns
        -------
        float
            nucleation driving force ce
            
        Notes
        -----
        The nucleation driving force depends on the stress invariants and
        material parameters, determining the nucleation threshold
        for the given loading conditions.
        """
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
