import numpy as np
import constants
import sklearn.manifold

def distance(strategy1, strategy2):
  shape = strategy1.shape
  bins = shape[-1] - 3
  call_idx = int(4 / 200 * bins)

  strat1_condensed = strategy1[..., constants.CHECK:]
  strat1_condensed[..., constants.CHECK] = strategy1[..., constants.FOLD] + strategy1[..., constants.CHECK]
  strat1_condensed[..., call_idx] += strategy1[..., constants.CALL]
  strat1_condensed /= np.sum(strat1_condensed)
  
  strat2_condensed = strategy2[..., constants.CHECK:]
  strat2_condensed[..., constants.CHECK] = strategy2[..., constants.FOLD] + strategy2[..., constants.CHECK]
  strat2_condensed[..., call_idx] += strategy2[..., constants.CALL]
  strat2_condensed /= np.sum(strat2_condensed)

  bhattacharyya_coeff = np.sqrt(strat1_condensed * strat2_condensed).sum()
  if bhattacharyya_coeff == 0:
    return None

  return -np.log(bhattacharyya_coeff)

def compute_mds(strategies):
  # Build distance matrix first
  strategy_tuples = list(strategies.items())
  dist_matrix = np.zeros((len(strategies), len(strategies)))
  for idx1,(agent1, strat1) in enumerate(strategy_tuples):
    for idx2,(agent2, strat2) in enumerate(strategy_tuples):
      dist = distance(strat1, strat2)
      if dist == None:
        print("Agents {0} and {1} have zero in common, throwing both".format(agent1, agent2))
        new_dict = dict(strategy_tuples)
        del new_dict[agent1]
        del new_dict[agent2]
        return compute_mds(new_dict)
      dist_matrix[idx1, idx2] = dist
      dist_matrix[idx2, idx1] = dist
    dist_matrix[idx1, idx1] = 0
    print("{}%".format(round(100 * idx1 / len(strategy_tuples), 2)))

  mds = sklearn.manifold.MDS(
    n_components=2,
    dissimilarity='precomputed'
  )
  embedding = mds.fit(dist_matrix).embedding_

  return {key: embedding[idx].tolist() for idx, (key, value) in enumerate(strategy_tuples)}