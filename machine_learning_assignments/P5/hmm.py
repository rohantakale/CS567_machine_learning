from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  """
  Forward algorithm

  Inputs:
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - alpha: A numpy array alpha[j, t-1] = P(Z_t = s_j, X_{1:t}=x_{1:t})
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])
  ###################################################
  # TODO 1.1 forward - alpha
  for t in range(N):
    # base case 
    if t == 0:
      alpha[:, 0] = np.multiply(pi, B[:,0])
        
    else:
      for s in range(S):
        alpha[s, t] = B[s, O[t]] * np.sum(np.multiply(alpha[:, t-1],A[:,s]))
        
  # print('alpha',alpha)
  ###################################################

  return alpha


def backward(pi, A, B, O):
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - beta: A numpy array beta[j, t-1] = P(X_{t+1:N}=x_{t+1:N} | Z_t = s_j)
  """
  S = len(pi)
  N = len(O)
  beta = np.zeros([S, N])
  ###################################################
  # TODO 1.1 backward - beta
  for t in range(N-1,-1,-1):
    if t == (N-1):
      beta[:,t] = 1
    else:
      for s in range(S):
        for j in range(S):
          beta[s,t] += beta[j,t+1] * A[s,j] * B[j,O[t+1]]
  
  # print('beta',beta)
  ###################################################
  
  return beta

def seqprob_forward(alpha):
  """
  Total probability of observing the whole sequence using the forward messages

  Inputs:
  - alpha: A numpy array alpha[j, t-1] = P(Z_t = s_j, X_{1:t}=x_{1:t})

  Returns:
  - prob: A float number of P(X_{1:N}=O)
  """
  prob = 0
  ###################################################
  # TODO 1.2 seqprob_forward
  prob = np.sum(alpha[:,-1])
  ###################################################
  
  return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward messages

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t-1] = P(X_{t+1:N}=x_{t+1:N} | Z_t = s_j)
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(X_{1:N}=O)
  """
  prob = 0
  ###################################################
  # TODO 1.2 seqprob_backward
  beta_pi = np.multiply(beta[:,0], pi)
  prob = np.sum(np.multiply(beta_pi, B[:,O[0]]))
  ###################################################
  
  return prob

def viterbi(pi, A, B, O):
  """
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path (in terms of the state index)
  """
  path = []
  ###################################################
  # TODO 1.3 Viterbi
  N = len(O)
  D = A.shape[0]
  # one - the probabilities of the most likely path
  # two - the states of the most likely path
  probs = np.zeros((D,N))
  state = np.zeros((D,N))


  #computing the most likely path at time t
  for t in range(N):
    if t == 0:
        probs[:,0] = np.multiply(pi,B[:,0])
    else:
        for j in range(D):
            probs[j,t] = np.max(np.multiply(probs[:,t-1],A[:,j]) * B[j,O[t]])
            state[j,t] = int(np.argmax(np.multiply(probs[:,t-1],A[:,j]) * B[j,O[t]]))
  # print('S',S)
  
  path.append(int(np.argmax(probs[:,N-1])))
  # backtracking
  for t in range(N-2,-1,-1):
    path.append(int(state[path[-1],t]))
  # print('path',path)
  ###################################################
  
  return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()