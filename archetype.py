import spams
import numpy as np
import torch
from timeit import default_timer as timer
from log_utils import get_logger

log = get_logger()
start = timer()

L = 1
log.info('ArchetypalAnalysis for ' + str(L) + ' layers')

X = torch.load('tensors/tensor'+str(L)+'.pt')
X = torch.t(X).numpy()
K = 32 # learns a dictionary with 32 elements
robust = True # use robust archetypal analysis or not, default parameter(True)
epsilon = 1e-3 # width in Huber loss, default parameter(1e-3)
computeXtX = True # memorize the product XtX or not default parameter(True)
stepsFISTA = 0 # 3 alternations by FISTA, default parameter(3)
# a for loop in FISTA is used, we stop at 50 iterations
# remember that we are not guarantee to descent in FISTA step if 50 is too small
stepsAS = 25 # 7 alternations by activeSet, default parameter(50)
randominit = True # random initilazation, default para

(Z,A,B) = spams.archetypalAnalysis(np.asfortranarray(X), returnAB= True, p = K, \
    robust = robust, epsilon = epsilon, computeXtX = computeXtX,  stepsFISTA = stepsFISTA , stepsAS = stepsAS, numThreads = -1)

print ('Evaluating cost function...')
alpha = spams.decompSimplex(np.asfortranarray(X), Z = Z, computeXtX = True, numThreads = -1)
xd = X - Z * alpha
R = np.sum(xd*xd)
print ("objective function: %f" %R)
print(Z.shape)
print(A.shape)
print(B.shape)
torch.save(Z, 'tensors/Z'+str(L)+'.pt')
torch.save(A, 'tensors/A'+str(L)+'.pt')
torch.save(B, 'tensors/B'+str(L)+'.pt')	

end = timer()
log.info('Time taken for archetypalAnalysis: ' + str(end - start) + 's')
	