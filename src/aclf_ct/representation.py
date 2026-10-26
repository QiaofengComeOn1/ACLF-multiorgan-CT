"""Feature-space projections and median-fixed response surfaces."""
import numpy as np
from sklearn.manifold import TSNE


def tsne_projection(X, seed=0, perplexity=30, learning_rate=200, iterations=1000, dimensions=3):
    """Project feature vectors with t-SNE."""
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or len(X)<=3 or not np.isfinite(X).all():
        raise ValueError('A finite matrix with more than three cases is required.')
    safe_upper=max(2.,float(len(X)-1))
    effective=max(2.,min(float(perplexity),safe_upper,max(5.,(len(X)-1)/3.)))
    return TSNE(n_components=int(dimensions),perplexity=effective,learning_rate=float(learning_rate),
        max_iter=int(iterations),init='pca',random_state=int(seed),metric='euclidean',verbose=0).fit_transform(X)


def conditional_response_surface(model,X,first,second,resolution=50):
    """Evaluate a selected feature pair while holding other inputs at cohort medians."""
    if first==second or first not in X or second not in X or resolution<2:
        raise ValueError('Two different available features and resolution >=2 are required.')
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Surface input matrix must be finite.')
    a=np.linspace(X[first].min(),X[first].max(),resolution)
    b=np.linspace(X[second].min(),X[second].max(),resolution)
    grid_a,grid_b=np.meshgrid(a,b)
    points=np.tile(X.median().to_numpy(),(resolution*resolution,1))
    points[:,X.columns.get_loc(first)]=grid_a.ravel()
    points[:,X.columns.get_loc(second)]=grid_b.ravel()
    predictions=model.predict_proba(points)[:,1].reshape(resolution,resolution)
    return grid_a,grid_b,predictions
