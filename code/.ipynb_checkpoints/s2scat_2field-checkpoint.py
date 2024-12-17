import copy
import healpy as hp
import jax
# JAX configuration
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import numpy as np
from jax import pmap
from matplotlib import pyplot as plt
import s2scat
from s2scat import compression
from s2scat.operators import spherical
from s2scat.utility import statistics, reorder
import s2wav


def flm_hp_to_2d_fast(flm_hp: np.ndarray, L: int) -> np.ndarray:
    """
    Converts a HEALPix-formatted array of spherical harmonic coefficients (flm) to a 2D array format.
    The resulting 2D array organizes these coefficients by their degree (ell) and order (m) for easier manipulation.

    Args:
        flm_hp (np.ndarray): Array of spherical harmonic coefficients in HEALPix format.
        L (int): Maximum degree of spherical harmonics to consider.

    Returns:
        np.ndarray: A 2D array where each row corresponds to a degree 'ell', and columns span 'm' values from -ell to ell.
    """
    # Initialize an empty 2D numpy array to hold flm data. The array will have complex data type to store spherical harmonics.
    flm_2d = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    
    for el in range(L):  # Loop over each degree 'el' from 0 to L-1.
        # Set the m=0 coefficient for the current 'el' in the center column of the 2D array.
        flm_2d[el, L - 1] = flm_hp[el]
        
        # Generate an array of m values from 1 to 'el'.
        m_array = np.arange(1, el + 1)
        # Compute the index in the HEALPix format array for each m value.f
        # This formula accounts for the unique indexing used by HEALPix to store coefficients.
        hp_idx = m_array * (2 * L - 1 - m_array) // 2 + el
        
        # Set the positive m values in the 2D array using the computed indices from the HEALPix array.
        flm_2d[el, L - 1 + m_array] = flm_hp[hp_idx]
        
        # For negative m values, set the corresponding coefficients in the 2D array.
        # This uses the conjugate of the corresponding positive m value coefficients, modified by (-1)^m.
        flm_2d[el, L - 1 - m_array] = (-1) ** m_array * np.conj(flm_hp[hp_idx])
    
    return flm_2d  # Return the 2D array containing reorganized spherical harmonic coefficients.


def pre_process_healpix_maps(healpix_maps, config = {}):
    """
    Pre-processes a list of HEALPix maps by converting them into spherical harmonic coefficients, 
    processing each map to compute wavelet transformed coefficients, and parallelizing the operations 
    across available computational devices.

    Args:
        healpix_maps (list of HEALPix maps): A list of spherical maps to process.

    Returns:
        tuple: A tuple of lists containing W coefficients, Mlm coefficients, and additional analysis 
               values from the wavelet transformation.
    """

    # Define a function to process individual maps to compute various spherical wavelet coefficients.
    def process_map(sphere_map, filters, precomputed_vals, config = {}):
        
        # Create full spherical harmonics if the map is real; otherwise, use the map as is.

        flm = spherical.make_flm_full(sphere_map, config['L']) if config['reality'] else sphere_map

        # Compute the initial analysis coefficients W for the spherical harmonics.
        W = spherical._first_flm_to_analysis(flm, config['L'], config['N'], config['J_min'], config['reality'], filters, precomputed_vals, config['recursive'])

        # Initialize lists to hold Mlm coefficients and other analysis values.
        Mlm_list = []
        value_list = []
        for j2 in range(config['J_min'], config['J_max'] + 1):
            # Determine the band limit for the current wavelet scale.
            Lj2 = s2wav.samples.wav_j_bandlimit(config['L'], j2, 2.0, True)

            # Compute the Mlm coefficients by transforming the absolute values of W.
            Mlm = spherical._forward_harmonic_vect(
                jnp.abs(W[j2 - config['J_min']]), j2, Lj2, config['J_min'], config['J_max'], config['reality'], precomputed_vals, config['recursive']
            )
            Mlm_list.append(Mlm)
            
            # Compute additional values for analysis using transformed coefficients.
            if j2 > config['J_min']:
                value = spherical._flm_to_analysis_vect(
                    Mlm, j2, Lj2, config['L'], config['N'], config['J_min'], j2 - 1, config['reality'], filters, precomputed_vals, config['recursive'],config['delta_j']
                )
                value_list.append(value)

        return W, Mlm_list, value_list

    # Function to manage the processing of maps in batches, depending on device availability.
    def run_in_batches(alm_jax, num_devices):
        num_maps = alm_jax.shape[0]
        W_list, Mlm_list, value_list = [], [], []

        for i in range(0, num_maps, num_devices):
            batch = alm_jax[i:i + num_devices]
            W_batch, Mlm_batch, value_batch = parallel_process(batch)
            W_list.extend(W_batch)
            Mlm_list.extend(Mlm_batch)
            value_list.extend(value_batch)

        return W_list, Mlm_list, value_list
    
    
    # Convert HEALPix maps to spherical harmonic coefficients focusing on positive m values.
    alm_maps = [flm_hp_to_2d_fast(hp.map2alm(map_data, lmax=config['L']-1), config['L'])[:, config['L']-1:] for map_data in healpix_maps]
    alm_jax = jnp.stack(alm_maps)
    

    # Configuration for processing that is shared across all devices.
    config_ = s2scat.configure(config['L'], config['N'], config['J_min'], config['reality'], config['recursive'], c_backend=False)
    filters, Q, precomputed_vals = config_

    # Retrieve the number of available GPUs/TPUs.
    num_devices = jax.local_device_count()



    # Parallel processing of maps using JAX's pmap.
    parallel_process = jax.pmap(
        lambda healpix_maps: process_map(healpix_maps, filters, precomputed_vals, config),
        axis_name='batch'
    )

    # Execute the batch processing and organize the results.
    W_array, Mlm_array, value_array = run_in_batches(alm_jax, num_devices)
    W_reordered = list(map(list, zip(*W_array)))
    M_reordered = list(map(list, zip(*Mlm_array)))
    V_reordered = list(map(list, zip(*[zip(*i) for i in value_array])))

    return W_reordered, M_reordered, V_reordered


def compute_statistics(W, M, val, Wa, Ma, vala, config):
    """
    Compute statistical measures S1, P00, and higher order covariances C01, C11 between two datasets.
    This function is configured to handle wavelet transformed data, compute basic and advanced statistical
    measures, and optionally compress results into isotropic coefficients.

    Args:
        W (list): List of wavelet coefficients for the first dataset.
        M (list): List of matrices/statistics for the first dataset.
        val (list): Additional values or weights for the first dataset.
        Wa (list): List of wavelet coefficients for the second dataset.
        Ma (list): List of matrices/statistics for the second dataset.
        vala (list): Additional values or weights for the second dataset.
        config (dict): Configuration dictionary with parameters such as L, N, J_min, J_max, etc.

    Returns:
        dict: Results containing computed statistical measures for each bin pair.
    """
    # Initialize the s2scat configuration based on provided settings.
    config_ = s2scat.configure(config['L'], config['N'], config['J_min'], config['reality'], config['recursive'], c_backend=False)
    filters, Q, precomputed_vals = config_
    
    results = dict()
    # Iterate through each bin of the first dataset.
    for bin1 in range(len(W)):
        W_ = W[bin1]
        M_ = M[bin1]
        val_ = val[bin1]
        # Iterate through each bin of the second dataset.
        for bin2 in range(len(Wa)):
            W1_ = Wa[bin2]
            M1_ = Ma[bin2]
            val1_ = vala[bin2]

            # Prepare lists to store the computed statistics.
            Nj1j2, Nj1j2b = [], []
            S1_, P00_ = [], []
            count = 0
            # Iterate over wavelet scales.
            for ii, j2 in enumerate(range(config['J_min'], config['J_max'] + 1)):
                Lj2 = s2wav.samples.wav_j_bandlimit(config['L'], j2, 2.0, True)

                # Compute modulus of the wavelet transform (SHT of |W|).
                Mlm = M_[ii]
                Mlm1 = M1_[ii]
                
                # Update S1 and P00 statistics for the current scale.
                S1_ = statistics.add_to_S1(S1_, Mlm, Lj2)
                P00_ = statistics.add_to_P00(P00_, W_[j2 - config['J_min']], Q[j2 - config['J_min']])
                
                # Compute higher order statistics Nj1j2 if above the minimum j2.
                if j2 > config['J_min']:
                    Nj1j2.append(val_[count])
                    Nj1j2b.append(val1_[count])
                    count += 1

            # Reorder and flatten nested lists to JAX arrays for computing covariances.
            Nj1j2_flat = reorder.nested_list_to_list_of_arrays(Nj1j2, config['J_min'], config['J_max'], config['delta_j'])
            Nj1j2_flat1 = reorder.nested_list_to_list_of_arrays(Nj1j2b, config['J_min'], config['J_max'], config['delta_j'])

            # Compute higher order covariances between the two datasets.
            C01, C11 = statistics.compute_C01_and_C11(Nj1j2_flat, Nj1j2_flat1, W1_, Q, config['J_min'], config['J_max'])

            # Optionally compress covariances to isotropic coefficients if requested in the config.
            if config.get('isotropic', False):
                C01, C11 = compression.C01_C11_to_isotropic(C01, C11, config['J_min'], config['J_max'])
            
            # Convert lists of statistics to 1D numpy arrays and reshape according to specified config.
            S1, S2, C01, C11 = reorder.list_to_array(S1_, P00_, C01, C11)
            S1 = np.mean(S1.reshape(-1, 2 * config['N'] - 1), axis=1)
            S2 = np.mean(S2.reshape(-1, 2 * config['N'] - 1), axis=1)
            
            # Store computed statistics for each bin pair.
            results['{0}_{1}'.format(bin1, bin2)] = {'S1':S1, 'S2':S2, 'C01':C01, 'C11':C11}
    
    return results


def compute_statistics_wrapper(maps1, maps2=None, config = {}):
    """
    Wrapper function to configure the environment, preprocess input maps, and compute statistics.
    This function allows the comparison of statistics between two sets of maps or within the same set.

    Args:
        maps1 (list): A list of HEALPix maps for the first dataset.
        maps2 (list, optional): A second list of HEALPix maps for comparison. If not provided,
                                statistics will be computed within the first set itself.


    Returns:
        dict: A dictionary containing computed statistics between all pairs of bins from the two datasets.
    """
    # Configure settings for spherical wavelet transformation and statistical analysis
    
    #config_ = s2scat.configure(config['L'], config['N'], config['J_min'], config['reality'], config['recursive'], c_backend=False)
    #filters, Q, precomps = config_
   # J_max = s2wav.samples.j_max(config['L'])
    #Q = spherical.quadrature(config['L'], config['J_min'])

    # Pre-process the first set of maps to get wavelet coefficients and associated statistics
    W1, M1, val1 = pre_process_healpix_maps(maps1,config)

    # Check if a second set of maps is provided, if not, use deep copies of the first set for self-comparison
    if maps2 is not None:
        W2, M2, val2 = pre_process_healpix_maps(maps2,config)
    else:
        # Use deep copies to ensure that modifications to one do not affect the other
        import copy
        W2 = copy.deepcopy(W1)
        M2 = copy.deepcopy(M1)
        val2 = copy.deepcopy(val1)

    # Compute and return the statistics between the two sets of wavelet coefficients and statistics
    return compute_statistics(W1, M1, val1, W2, M2, val2, config)


########################################################
#                Load data 
########################################################

config = dict()
config['nside'] = 1024
config['L'] = config['nside']*3           # Spherical harmonic bandlimit.
config['lam'] = 2.0            # 
config['N'] = 2               # Azimuthal bandlimit (directionality). 2N−1 directions
config['J_max'] = s2wav.samples.j_max(config['L'])
config['J_min'] = config['J_max'] -5
config['isotropic'] = True
config['recursive'] = True # Input signal is real.
config['reality'] = True  # Use the fully precompute transform. (True: slower but more memory efficient)- 
config['delta_j'] = 1
      
########################################################
#                run on  maps
########################################################
maps = [np.random.normal(0,1,config['nside']**2*12),
        np.random.normal(0,1,config['nside']**2*12),
        np.random.normal(0,1,config['nside']**2*12),
        np.random.normal(0,1,config['nside']**2*12)]
    
results = compute_statistics_wrapper(maps,maps, config)
