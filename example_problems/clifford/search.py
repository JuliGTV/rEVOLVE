
import numpy as np
import time
import itertools as iter

class paramObj:
    pass

def search(U):
    params = paramObj()
    params.mode = 'GL'
    params.method = 'greedy'
    params.minDepth = False
    params.hv = 1 ## vector
    params.hi = 1 ## include inverse
    params.ht = 1 ## include transpose
    params.hl = 1 ## log of cols 1 or sums 0
    params.hr = 3 # scaling factor for heuristic
    params.wMax = 10

    n,gateCount,depth,procTime,check,circ = synth_GL(U)

    return n,gateCount,depth,procTime,check,circ

# ==== IMPORTED FUNCTIONS FROM CLIFFORDOPT PACKAGE ====

# From common.py
def currTime():
    '''Return current time'''
    return time.time()

def ZMatZeros(s):
    '''Return integer array of zeros of length/shape s'''
    return np.zeros(s,dtype=int)

def ZMatI(n):
    '''Identity n x n integer matrix'''
    return np.eye(n,dtype=int)

def ZMat(A,n=None):
    '''Create an integer numpy array. If n is set, ensure that the row length is n.'''
    if typeName(A) in ['set','range']:
        A = list(A)
    if typeName(A) != 'ndarray' or A.dtype != int:
        A = np.array(A,dtype=int)
    if n is not None:
        s = list(A.shape)
        if s[-1] == 0:
            A= np.empty((0,n),dtype=int)
    return A

def typeName(val):
    '''Return the name of the type of val in text form.'''
    return type(val).__name__

def ixRev(ix):
    ## return indices to restore original column order
    ## input: ix - permutation of [0..n-1]
    ## output: ixR such that ix[ixR] = [0..n-1]
    n = max(ix) + 1
    ixR = ZMatZeros(n)
    ixR[ix] = range(len(ix))
    return ixR

def matSum(A):
    '''Sum of all elements in A'''
    return np.sum(A)

def matColSum(A):
    '''Sum of columns in A'''
    return np.sum(A, axis=0)

def vecJoin(*args):
    '''Join vectors'''
    return np.concatenate(args)

def bin2Set(v):
    '''Convert binary vector to a list of indices such that v[i] !=0'''
    v = np.ravel(v)
    return list(np.nonzero(v)[0])

def nonDecreasing(w):
    '''Check whether vector w is non-decreasing'''
    for i in range(1,len(w)):
        if w[i] < w[i-1]:
            return False 
    return True

# From CliffordOps.py
def symShape(U):
    m,n = U.shape
    return m//2, n//2

def binMatInv(A):
    H, U = getHU(A,2)
    return U

def permMat2ix(P):
    ## Check if this is a permutation matrix
    P = ZMat(P)
    if np.any(matColSum(P.T)!=1):
        return False
    m,n = P.shape
    ix = ZMatZeros(m)
    A = np.nonzero(P)
    for i in range(len(A[1])):
        ix[i] = A[1][i]
    return ix

def symR2(U):
    '''return an nxn binary matrix such that S[i,j] = 1 if F_ij is invertible or 0 otherwise'''
    m,n = symShape(U)
    ## we calculate the determinant in parallel: U_XX U_ZZ + U_XZ U_ZX
    UR2 = (U[:m,:n] & U[m:,n:])
    UR2 ^= (U[:m,n:] & U[m:,:n])
    return UR2

def Fmat(U,i,j):
    '''Return F-matrix: U_{i,j} & U_{i,j+n}\\U_{i+n,j} & U_{i+n,j+n}'''
    m,n = symShape(U)
    F = ZMatZeros((2,2))
    for r in range(2):
        for c in range(2):
            F[r,c] = U[i + m*r, j + n*c]
    return F

def isIdPerm(ix):
    '''return true if ix is an identity permutation'''
    return nonDecreasing(ix)

# From NHow.py (we need to include getHU function)
def getHU(A, N, nC=None, retPivots=False, tB=1):
    '''Hermite Normal Form of A mod N with transformation matrix U such that H = U @ A'''
    if nC is None:
        nC = len(A.T)
    A = ZMat(A)
    m, n = A.shape
    A = np.column_stack([A, ZMatI(m)])
    H, pivots = getH(A, N, nC, retPivots=True, tB=tB)
    U = H[:, nC:]
    H = H[:, :nC]
    if retPivots:
        return H, U, pivots
    return H, U

def getH(A, N, nC=None, retPivots=False, tB=1):
    '''Hermite Normal Form of A mod N'''
    if nC is None:
        nC = len(A.T)
    A = ZMat(A)
    m, n = A.shape
    if tB > 1:
        A = ZMatBlockify(A, tB)
        m, n = A.shape
    r, c = 0, 0
    pivots = []
    while r < m and c < nC:
        # Find pivot
        pivot_row = None
        for i in range(r, m):
            if np.any(A[i, c:c+tB] != 0):
                pivot_row = i
                break
        
        if pivot_row is not None:
            # Swap rows if needed
            if pivot_row != r:
                A[[r, pivot_row]] = A[[pivot_row, r]]
            
            # Find leading element
            pivot_col = c
            for j in range(c, min(c+tB, nC)):
                if A[r, j] != 0:
                    pivot_col = j
                    break
            
            pivots.append(pivot_col)
            
            # Eliminate other rows
            for i in range(m):
                if i != r and A[i, pivot_col] != 0:
                    A[i] = (A[i] - A[r]) % N
            r += 1
        c += tB
    
    if retPivots:
        return A, pivots
    return A

def ZMatBlockify(A, tB):
    '''Helper for blocked operations'''
    return A



def GLHeuristic(U,params):
    '''calculate heuristics - vector and scalar - for symplectic matrices'''
    m,n = symShape(U)
    if params.hi == 0:
        U = U[:m,:n]
    sA = vecJoin(matColSum(U),matColSum(U.T)) if params.ht else matColSum(U)
   
    w = tuple(sorted(sA))
    Ls = len(sA)
    h =  matSum(np.log(sA))/Ls if params.hl else (matSum(U)/len(U) - 1)
    return params.hr * h, w

def SpHeuristic(U,params):
    '''calculate heuristics - vector and scalar - for symplectic matrices'''
    hi,ht,hl,hr = params.hi,params.ht,params.hl,params.hr
    m,n = symShape(U)
    ## Invertible 2x2 matrices
    UR2 = symR2(U)
    ## All zero 2x2 matrices
    UR0 = symR0(U)
    ## Rank 1 2x2 matrices - not U1 and not U2
    UR1 = symR1(UR2,UR0)
    c1 = vecJoin(matColSum(UR1),matColSum(UR1.T)) if ht else matColSum(UR1)
    c2 = vecJoin(matColSum(UR2),matColSum(UR2.T)) if ht else matColSum(UR2)
    if hl:
        h = matSum(np.log(c1 + c2))/len(c1)
    else:
        h = (matSum(UR1) + matSum(UR2))/n - 1
    return hr * h, tuple(sorted(c2 * n + c1))

def symR0(U):
    '''return an nxn binary matrix such that S[i,j] = 1 if F_ij is zero'''
    m,n = symShape(U)
    ## Flip 0 and 1
    U = 1 ^ U
    ## all zero entries have 1 in all four of U_XX U_XZ U_ZX U_ZZ so multiply together these matrices
    UR0 = (U[:m,:n] & U[m:,n:]) & (U[:m,n:] & U[m:,:n])
    return UR0

def symR1(UR2,UR0):
    '''return an nxn binary matrix such that S[i,j] = 1 if F_ij is rank 1'''
    ## S_ij=1 if either F_ij is rank 2 or rank 0
    UR1 = (UR2 ^ UR0)
    ## Flipping 0 and 1 results in F_ij rank 1
    UR1 ^= 1
    return UR1

def GLOptions(A,allOpts=False):
    '''CNOT gate options for GL reduction'''
    m,n = symShape(A)
    U = A[:m,:n]
    if allOpts:
        return [('CX',(i,j)) for i in range(n) for j in range(n) if i!=j]
    ## dot product of columns with columns - non-zero elements have overlap and so are in the list
    iList,jList = np.nonzero((U.T @ U)) 
    ## exclude those along the diagonal
    return [('CX',(i,j)) for (i,j) in zip(iList,jList) if i != j]

def SpOptions(U):
    '''Transvection gate options for symplectic reduction'''
    m,n = symShape(U)
    ijList = set()
    UR2 = symR2(U)
    UR0 = symR0(U)
    UR1 = symR1(UR2,UR0)
    for i in range(m):
        R2 = bin2Set(UR2[i])
        R1 = bin2Set(UR1[i])
        L = len(R2)
        for j in range(L-1):
            for k in range(j+1,L):
                ijList.add((R2[j],R2[k]))
        for j in R2:
            for k in R1:
                ijList.add((j,k))
    vList = {(a % 2,b%2,a//2,b//2) for a in range(1,4) for b in range(1,4)}
    return {(v,ij) for v in vList for ij in ijList}

def applyOp(U,myOp,update=False):
    '''apply op to U - update U if update=True'''
    m,n = symShape(U)
    opType,qList = myOp
    if not update:
        U = U.copy()
    if opType == 'QPerm':
        ix = ZMat(qList)
        ix = vecJoin(ix,n + ix)
        U = U[:,ix]
    elif opType == "CX":
        (i,j) = qList
        U[:,j] ^= U[:,i]
        U[:,n+i] ^= U[:,n+j]
    return U

def opListLayers(opList):
    '''split opList into layers to calculate circuit depth'''
    layers = []
    for opType,qList in opList:
        if opType not in {'QPerm','SWAP'} and len(qList) > 1:
            L = len(layers)
            i = L
            qList = set(qList)
            while i > 0 and len(qList.intersection(layers[i-1])) == 0:
                i = i-1
            if i == L:
                layers.append(qList)
            else:
                layers[i].update(qList)
    return layers

def entanglingGateCount(opList):
    '''count number of entangling gates in list of operataors opList'''
    c = 0
    for opName,qList in opList:
        if isEntangling(opName,qList):
            c += 1
    return c

def isEntangling(opName, qList):
    '''check if op is entangling'''
    ## any gate actingh on more than one qubit which is not a SWAP or QPerm
    return len(qList) > 1 and opName != 'SWAP' and opName != "QPerm"

def opList2str(opList,ch="\n"):
    '''convert oplist to string rep'''
    pauli_list = ['I','X','Z','Y']
    temp = []
    for opName,qList in opList:
        if typeName(opName) in ('tuple','ndarray'):
            opName = ZMat2str(opName)
        opName = opName.replace(" ","")
        qStr = ",".join([str(q) for q in qList])
        temp.append(f'{opName}:{qStr}')
    return ch.join(temp)

def ZMat2str(A,N=None):
    '''Return string version of integer matrix A.'''
    if np.size(A) == 0:
        return ""
    S = np.char.mod('%d', A)
    sep = ""
    if N is None:
        N = np.amax(A) + 1
    if N > 10:
        Nw= len(str(N-1))
        S = np.char.rjust(S,Nw)
        sep = " "
    return sep.join(S)

def symTest(U,opList):
    '''check that opList circuit is equivalent to symplectic matrix U'''
    m,n = symShape(U)
    U2 = opList2sym(opList,n)
    return binMatEq(U,U2)

def opList2sym(opList,n):
    '''convert opList to 2n x 2n binary symplectic matrix '''
    return applyOpList(opList,ZMatI(2*n),True)

def applyOpList(opList,A,update=False):
    '''apply list of operations'''
    if type(opList) == tuple:
        opList = [opList]
    if not update:
        A = A.copy()
    for myOp in opList:
        A = applyOp(A,myOp,True)
    return A

def binMatEq(A,B):
    return (matSum(A ^ B) == 0)

def opListInv(opList):
    '''return inverse of opList'''
    temp = []
    for (opType,qList) in reversed(opList):      
        qList = ZMat(qList) 
        if opType == 'QPerm':
            qList = ixRev(qList)
        elif opType == 'HS':
            opType = 'SH'
        elif opType == 'SH':
            opType = 'HS'
        temp.append((opType,tuple(qList)))
    return temp

# Single-qubit Clifford functionality
SQC_tostr = {'1001':'I', '0110':'H','1101':'S','1011':'HSH','1110':'HS','0111':'SH'}
SQC_fromstr = {v : np.reshape([int(i) for i in k],(2,2)) for k,v in SQC_tostr.items()}

def SQC2str(A):
    '''convert a 2x2 single qubit Clifford matrices to opType'''
    global SQC_tostr
    return SQC_tostr[ZMat2str(A.ravel())] 

def CList2opList(UC):
    '''convert list of 2x2 SQC matrices to opList'''
    temp = []
    ## dict for single-qubit Cliffords
    for i in range(len(UC)):
        c =  SQC2str(UC[i])
        ## don't add single-qubit identity operators
        if c != 'I':
            temp.append((c,[i]))
    return temp

# ============================================================


def synth_GL(U,params):
    '''synthesis - GL matrix for CNOT circuits'''
    ## convert GL to Symplectic matrix and run synth_main
    return synth_main(symCNOT(U),params)


def synth_main(U,params,qc=None):
    '''main synthesis function - calls various optimsation functions using options in params. U is a symplectic matrix, qc is a Qiskit circuit'''

    m,n = symShape(U)
    ## starting time
    sT = currTime()

    ############################################################
    ## Paper algorithms
    ############################################################

    ## Greedy Algorithm
    if params.method == 'greedy':
        opList, UC = csynth_greedy(U,params)
        opList = mat2SQC(UC) + opList

    depth = len(opListLayers(opList))
    gateCount = entanglingGateCount(opList)
    procTime = currTime()-sT
    circ = opList2str(opList,ch=" ")
    MWalgs = ['optimal','volanto','greedy','astar','CNOT_optimal','CNOT_gaussian','CNOT_Patel','CNOT_greedy','CNOT_astar','CNOT_depth']
    if params.method in MWalgs:
        check = symTest(U,opList)
    else:
        check = ""
    return n,gateCount,depth,procTime,check,circ


def csynth_greedy(A,params):
    '''Decomposition of symplectic matrix A into 2-transvections, SWAP and single-qubit Clifford layers'''
    mode = params.mode
    m,n = symShape(A)
    A = A.copy()
    opList = []
    hix = 1 if params.hv else 0
    h,w = GLHeuristic(A,params) if mode=='GL' else  SpHeuristic(A,params)
    hMin,hLast = None,None
    currWait = 0
    dMax = 10000
    while h > 0.00001:
        gateOpts = GLOptions(A,False) if mode == 'GL' else  SpOptions(A)
        dhMin,BMin = None,None
        for myOp in gateOpts:
            B = applyOp(A,myOp) 
            h,w = GLHeuristic(B,params) if mode=='GL' else  SpHeuristic(B,params)
            hB = (w,h,myOp) if params.hv else (h,w,myOp)
            if params.minDepth:
                d = len(opListLayers(opList + [myOp])) 
                if hLast is not None and hB > hLast:
                    d += dMax
            else:
                d = 0
            dhB = (d,hB)
            if (dhMin is None) or (dhMin > dhB):
                dhMin,BMin = dhB,B
        # print(dhMin)
        hLast = dhMin[1]
        h = hLast[hix]
        opList.append(hLast[-1])
        A = BMin
        if hMin is None or hLast < hMin:
            currWait = 0
            hMin = hLast
        else:
            currWait += 1
        if (params.wMax > 0 and currWait > params.wMax):
            return [],np.arange(n),[]
    opList = opListInv(opList)
    return opList,A




def symCNOT(U):
    '''CNOT circuit from binary invertible matrix U'''
    m,n = U.shape
    U2 = ZMatZeros((2*n,2*n))
    U2[:n,:n] = U
    Uinv = binMatInv(U.T)
    U2[n:,n:] = Uinv
    return U2


def ZMatZeros(s):
    '''Return integer array of zeros of length/shape s'''
    return np.zeros(s,dtype=int)


def mat2SQC(UC):
    '''convert matrix UC to opList representing qubit permutation and list of single-qubit cliffords'''
    UR2 = symR2(UC)
    ix = permMat2ix(UR2)
    ixR = ixRev(ix)
    ## extract list of single-qubit cliffords
    CList =  [Fmat(UC,i,ix[i]) for i in ixR]
    temp =  CList2opList(CList)
    ## check if we have the trivial permutation - if not append a QPerm operator
    if not isIdPerm(ixR):
        temp = [('QPerm', ixR)] + temp
    return temp