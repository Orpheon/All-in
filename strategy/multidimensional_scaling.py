import numpy as np
import constants
import sklearn.manifold
import sklearn.cluster
import pandas as pd

def condense(strategy):
  shape = strategy.shape
  bins = shape[-1] - 3
  call_idx = int(4 / 200 * bins)
  
  strat_condensed = strategy[..., constants.CHECK:]
  strat_condensed[..., constants.CHECK] = strategy[..., constants.FOLD] + strategy[..., constants.CHECK]
  strat_condensed[..., call_idx] += strategy[..., constants.CALL]
  strat_condensed /= np.sum(strat_condensed)

  return strat_condensed

def distance_bhattacharyya(strategy1, strategy2):
  strat1_condensed = condense(strategy1)
  strat2_condensed = condense(strategy2)

  bhattacharyya_coeff = np.sqrt(strat1_condensed * strat2_condensed).sum()
  if bhattacharyya_coeff == 0:
    return None

  return -np.log(bhattacharyya_coeff)

def distance_euclidean(strategy1, strategy2):
  strat1_condensed = condense(strategy1)
  strat2_condensed = condense(strategy2)

  distance = np.linalg.norm(strat1_condensed - strat2_condensed)
  return distance

def compute_mds(strategies, verbose=True):
  # Build distance matrix first
  strategy_tuples = list(strategies.items())
  dist_matrix = np.zeros((len(strategies), len(strategies)))
  for idx1,(agent1, strat1) in enumerate(strategy_tuples):
    for idx2,(agent2, strat2) in enumerate(strategy_tuples):
      # dist = distance_bhattacharyya(strat1, strat2)
      dist = distance_euclidean(strat1, strat2)
      if dist == None:
        if verbose:
          print("Agents {0} and {1} have zero in common, throwing both".format(agent1, agent2))
        new_dict = dict(strategy_tuples)
        del new_dict[agent1]
        del new_dict[agent2]
        return compute_mds(new_dict)
      dist_matrix[idx1, idx2] = dist
      dist_matrix[idx2, idx1] = dist
    dist_matrix[idx1, idx1] = 0
    if verbose:
      print("MDS: {}%".format(round(100 * idx1 / len(strategy_tuples), 2)))

  mds = sklearn.manifold.MDS(
    n_components=2,
    dissimilarity='precomputed'
  )
  embedding = mds.fit(dist_matrix).embedding_

  return {key: embedding[idx].tolist() for idx, (key, value) in enumerate(strategy_tuples)}

def kmeans(strategies, MAX_N=12, verbose=True):
  strategy_tuples = list(strategies.items())
  results = pd.DataFrame(index=[x[0] for x in strategy_tuples])
  flattened_strats = np.array([x[1].flatten() for x in strategy_tuples])
  for n in range(2, MAX_N):
    if verbose: print(n, "clusters on", MAX_N - 1)
    kmeans = sklearn.cluster.KMeans(n_clusters=n)
    clusters = kmeans.fit_predict(X=flattened_strats)
    results[n] = clusters

  print(results)
  return results