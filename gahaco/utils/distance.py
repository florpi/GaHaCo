import numpy as np
def periodic_distance(df, boxsize: float) -> np.array:
    a = np.array([df['x_dmo'], df['y_dmo'], df['z_dmo']])
    b = np.array([df['x_hydro'], df['y_hydro'], df['z_hydro']])

    bounds = boxsize * np.ones(3)
    min_dists = np.min(np.dstack(((a - b) % bounds, (b - a) % bounds)), axis=2)
    dists = np.sqrt(np.sum(min_dists ** 2, axis=1))
    return dists[0]


def distance_criteria(df, threshold):
    boxsize = 300.
    df['displacement'] = df.apply(lambda x: periodic_distance(x, boxsize), axis=1)
    print(f'Using threshold = {threshold}')
    df = df[df['displacement'] < threshold]
    return df
