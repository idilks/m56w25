import numpy as np


def lu_gewp(B):
    """Given an invertible matrix B, attempts to produce the LU factorization of B. Recall that this may fail
    if there is a zero pivot!
    """

    A = B.copy() # make a copy
    n = A.shape[1] # size
    L = np.eye(n) # array for storing L
    U = np.zeros((n,n)) # array for storing U
    growth_max_init = np.amax(np.abs(A))
    growth_max_all_time = np.amax(np.abs(A))

    # Recall that there will be n-1 steps
    for j in range(n-1):

        # Look at the jth column of A, compute the needed multipliers
        m = -A[j+1:,j]/A[j,j]

        # Stash the (negative) multipliers in result L!
        L[j+1:,j] = -m

        # Apply elementary matrix to 
        # Use the expression M_j = I + m e_j^T
        for l in range(j,n):
            A[j+1:,l] += A[j,l]*m

        # Update tracker for the growth factor
        new_max = np.amax(np.abs(A))
        if new_max > growth_max_all_time:
            growth_max_all_time = new_max

    # Compute growth factor at the end
    growth_factor = growth_max_all_time/growth_max_init

    return L, np.triu(A), growth_factor





def lu_gepp(B):
    """Given an invertible matrix B, produce the PB = LU factorization of B using GEPP.
    """

    A = B.copy() # make a copy
    n = A.shape[1] # size
    L = np.eye(n) # array for storing L
    U = np.zeros((n,n)) # array for storing U
    P = np.eye(n)
    growth_max_init = np.amax(np.abs(A))
    growth_max_all_time = np.amax(np.abs(A))

    # Recall that there will be n-1 steps
    for j in range(n-1):

        # Look at the jth column of A, decide whether to swap any rows
        current_row_idx = j
        candidates = np.abs(A[:,j])
        candidates[:j] = 0.0 # set to zero so these won't get chosen
        max_row_idx = np.argmax(candidates)
        if max_row_idx != current_row_idx:
            A[[max_row_idx, current_row_idx]] = A[[current_row_idx, max_row_idx]]
            P[[max_row_idx, current_row_idx]] = P[[current_row_idx, max_row_idx]]
            
            # Also apply the swap to the previous m's
            if j > 0:
                for k in range(j):
                    L[[max_row_idx, current_row_idx],k] = L[[current_row_idx, max_row_idx],k]
            else:
                pass

        else:
            pass

        # Look at the jth column of A, compute the needed multipliers
        m = -A[j+1:,j]/A[j,j]

        # Stash the (negative) multipliers in result L!
        L[j+1:,j] = -m

        # Apply elementary matrix using the expression M_j = I + m e_j^T
        for l in range(j,n):
            A[j+1:,l] += A[j,l]*m

        # Update tracker for the growth factor
        new_max = np.amax(np.abs(A))
        if new_max > growth_max_all_time:
            growth_max_all_time = new_max

    # Compute growth factor at the end
    growth_factor = growth_max_all_time/growth_max_init

    return P, L, np.triu(A), growth_factor




def lu_gecp(B):
    """Given an invertible matrix B, produce the PBQ = LU factorization of B using GECP.
    """

    A = B.copy() # make a copy
    n = A.shape[1] # size
    L = np.eye(n) # array for storing L
    U = np.zeros((n,n)) # array for storing U
    P = np.eye(n)
    Q = np.eye(n)
    growth_max_init = np.amax(np.abs(A))
    growth_max_all_time = np.amax(np.abs(A))

    # Recall that there will be n-1 steps
    for j in range(n-1):

        # Look at the jth column of A, decide whether to swap any rows
        current_row_idx = j
        current_col_idx = j
        candidates = np.abs( A.copy() )
        candidates[:,:j] = 0.0 # set to zero so these won't get chosen
        candidates[:j,:] = 0.0 # set to zero so these won't get chosen
        max_flat_index = np.argmax(candidates)
        max_row_idx, max_col_idx = np.unravel_index(max_flat_index, candidates.shape)
        if max_row_idx != current_row_idx:
            A[[max_row_idx, current_row_idx]] = A[[current_row_idx, max_row_idx]]
            P[[max_row_idx, current_row_idx]] = P[[current_row_idx, max_row_idx]]
            
            # Also apply the swap to the previous m's
            if j > 0:
                for k in range(j):
                    L[[max_row_idx, current_row_idx],k] = L[[current_row_idx, max_row_idx],k]
            else:
                pass

        else:
            pass

        if max_col_idx != current_col_idx:
            A[:,[max_col_idx, current_col_idx]] = A[:,[current_col_idx, max_col_idx]]
            Q[:,[max_col_idx, current_col_idx]] = Q[:,[current_col_idx, max_col_idx]]

        # Look at the jth column of A, compute the needed multipliers
        m = -A[j+1:,j]/A[j,j]

        # Stash the (negative) multipliers in result L!
        L[j+1:,j] = -m

        # Apply elementary matrix using the expression M_j = I + m e_j^T
        for l in range(j,n):
            A[j+1:,l] += A[j,l]*m

        # Update tracker for the growth factor
        new_max = np.amax(np.abs(A))
        if new_max > growth_max_all_time:
            growth_max_all_time = new_max

    # Compute growth factor at the end
    growth_factor = growth_max_all_time/growth_max_init

    return P, Q, L, np.triu(A), growth_factor






# def lu_gerp(B):
#     """Given an invertible matrix B, produce the PBQ = LU factorization of B using GERP.
#     """

#     # Bonus for you to do!

#     return P, Q, L, np.triu(A), growth_factor







def cholfac(A):
    """Given a SPD matrix A, computes L in the Cholesky factorization A = L L^T.
    """

    n = A.shape[0] 
    L = np.zeros((n,n))

    # Do the first step manually
    Lsub = np.sqrt(A[0,0])
    L[0,0] = Lsub

    Asub = A[:2, :2]
    d = Asub[0,-1] # d
    gamma = Asub[-1,-1] # gamma

    r = d/Lsub
    rho = np.sqrt(gamma - np.dot(r,r))

    Lsub = np.asarray([ [ Lsub, 0 ], [r, rho] ] )

    # Now loop over the rest
    for i in range(2,n):
        
        Asub = A[:i+1, :i+1]
        d = Asub[:-1, -1]
        gamma = Asub[-1, -1]

        r = fsub(Lsub, d)
        rho = np.sqrt(gamma - np.dot(r,r))

        Lsub = np.vstack([ Lsub, r.T ])
        tmp = np.zeros(Lsub.shape[0])
        tmp[-1] = rho
        Lsub = np.hstack([ Lsub, tmp[:,None] ])

    return Lsub