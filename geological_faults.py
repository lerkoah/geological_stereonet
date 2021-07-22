import numpy as np
import matplotlib.pyplot as plt
import mplstereonet

from sklearn.metrics import silhouette_score

from matplotlib import cm
from matplotlib.colors import ListedColormap   
    

class SphericalKmeans():
    
    def __init__(self, num=2):
        self.centers = None
        self.covs = None
        self.inertial_ = None
        self.dists = None
        
        self.num = num

    def fit(self, *args, **kwargs):
        """
        Find centers of multi-modal clusters of data using a kmeans approach
        modified for spherical measurements.

        Parameters
        ----------

        *args : 2 or 3 sequences of measurements
            By default, this will be expected to be ``strike`` & ``dip``, both
            array-like sequences representing poles to planes.  (Rake measurements
            require three parameters, thus the variable number of arguments.) The
            ``measurement`` kwarg controls how these arguments are interpreted.

        num : int
            The number of clusters to find. Defaults to 2.

        bidirectional : bool
            Whether or not the measurements are bi-directional linear/planar
            features or directed vectors. Defaults to True.

        tolerance : float
            Iteration will continue until the centers have not changed by more
            than this amount. Defaults to 1e-5.

        measurement : string, optional
            Controls how the input arguments are interpreted. Defaults to
            ``"poles"``.  May be one of the following:

                ``"poles"`` : strikes, dips
                    Arguments are assumed to be sequences of strikes and dips of
                    planes. Poles to these planes are used for analysis.
                ``"lines"`` : plunges, bearings
                    Arguments are assumed to be sequences of plunges and bearings
                    of linear features.
                ``"rakes"`` : strikes, dips, rakes
                    Arguments are assumed to be sequences of strikes, dips, and
                    rakes along the plane.
                ``"radians"`` : lon, lat
                    Arguments are assumed to be "raw" longitudes and latitudes in
                    the stereonet's underlying coordinate system.

        Returns
        -------

        centers : An Nx2 array-like
            Longitude and latitude in radians of the centers of each cluster.
        """
        lon, lat = mplstereonet.analysis._convert_measurements(args, kwargs.get('measurement', 'poles'))
        bidirectional = kwargs.get('bidirectional', True)
        tolerance = kwargs.get('tolerance', 1e-5)

        points = lon, lat
        dist = lambda x: mplstereonet.stereonet_math.angular_distance(x, points, bidirectional)
        

        center_lon = np.random.choice(lon, self.num)
        center_lat = np.random.choice(lat, self.num)
        self.centers = np.column_stack([center_lon, center_lat])

        while True:
            self.dists = np.array([dist(item) for item in self.centers]).T
            closest = self.dists.argmin(axis=1)

            new_centers = []
            intertial_tmp = []
            for i in range(self.num):
                mask = closest == i
                _, vecs = mplstereonet.analysis.cov_eig(lon[mask], lat[mask], bidirectional)
                new_centers.append(mplstereonet.stereonet_math.cart2sph(*vecs[:,-1]))
#                 print(dists[mask, i].shape)
#                 intertial_tmp.append((self.dists[mask, i]**2).sum())

            if np.allclose(self.centers, new_centers, atol=tolerance):
                break
            else:
                self.centers = new_centers
#                 self.inertial_ = np.array(intertial_tmp).sum()

        return self.centers

def plot_polar_with_kde2(strike, dip, c='k', cmap='Blues', arg_dipdir=True, ax=None, fig=None, show_points=True, alpha_pole=0.1):
    if ax is None:
        fig = plt.figure(figsize=(13,8))
        ax = fig.add_subplot(111, projection='stereonet')
        
        
    if arg_dipdir:
        strike = strike-90
        
    if show_points:
        ax.pole(strike, dip, c=c, markersize=1, zorder=3, alpha=alpha_pole)
    ax.rotation
    cax = ax.density_contourf(strike, dip, measurement='poles', cmap=cmap, levels=25, zorder=1, alpha=0.3)
#     fig.colorbar(cax)
    
    return fig, ax


def silhouette_routine(strikes, dip, kmax=10):
    modelsList = []
    silhouettes = []
    K = np.arange(2,10)
    for k in K:
        kmeans = SphericalKmeans(k)
        kmeans.fit(strikes, dip, measurement='poles')

        # Compute silhouettes score
        lon, lat = mplstereonet.analysis._convert_measurements((strikes, dip), 'poles')
        Mdists = np.array([mplstereonet.angular_distance(x, (lon, lat)) for x in zip(lon, lat)])
        silhouettes.append(silhouette_score(Mdists, kmeans.dists.argmin(axis=1)))

        # Save model
        modelsList.append(kmeans)
    
    return modelsList, np.array(silhouettes), K


cmaps = ['Blues', 'Reds', 'Greens', 'Purples']
cmaps_new = []
for c in cmaps:
    viridis = cm.get_cmap(c, 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 0])
    newcolors[:50, :] = white
    newcmp = ListedColormap(newcolors)
    cmaps_new.append(newcmp)
colors = ['b', 'r', 'g', 'm']

def _plotting_model(km, dipdir, dip):
    clusterAssign = km.dists.argmin(axis=1)

    fig = plt.figure(figsize=(13,8))
    ax = fig.add_subplot(111, projection='stereonet')
    
    
    idxs = dipdir.index
    for i in range(km.num):
        mask_tmp = idxs[clusterAssign==i]
        plot_polar_with_kde2(dipdir.loc[mask_tmp], dip.loc[mask_tmp], 
                             cmap=cmaps_new[i], c=colors[i], ax=ax, fig=fig)
    ax.grid(True)
    
    return fig, ax
