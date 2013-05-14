def invmatgen_lidar(P_in):
    import numpy as np

    beta_mat = np.diag((P_in.beta_m + P_in.beta_p))

    A_bar = np.linalg.inv(beta_mat)*P_in.backsub_vals

    A = np.diag(A_bar)

    return A

def invmatgen_raman(P_in):
    import numpy as np

    alpha_p0_mat = np.diag(P_in.alpha_p0)

    A_bar = np.linalg.inv(alpha_p0_mat)*P_in.backsub_vals

    A = np.diag(A_bar)

    return A

def raman_error(P_est,P_0):
    """ Calculate the residual from an estimated frofile given the known original
        this is for raman profiles, so the residual is calculated on the
        particulate extinction ratio at laser wavelength alpha_p0

    """

    try:
        residual = P_est.backsub_vals-P_0.backsub_vals       
    except AttributeError:
        residual = P_est.vals-P_0.vals

    error = P_est.alpha_p0 - P_0.alpha_p0

    P_est.res = residual
    P_est.err = error

    return P_est


