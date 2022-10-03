import numpy as np
from scipy.stats import multivariate_normal
import networkx as nx
import networkx.algorithms.community as nx_comm

def UpdateBlockParameters (ExpMat,Network):
	#This function updates block model parameters under the SBM model
	#Inputs: ExpMat: the n x k matrix of expectations
		#Network: adjacency matrix
	
	NewBlock = np.zeros(ExpMat.shape)
	for q in range(NewBlock[0]):
		for l in range(q,):
			denom = 0
			num = 0
			for i in range(Network.shape[0]):
				for j in range(i,Network.shape[0]):
					num += ExpMat[i,q]*ExpMat[j,l]*Network[i,j]
					denom += np.max(0.00000001, ExpMat[i,q]*ExpMat[j,l])
			NewBlockM[q,l] = num/denom
			NewBlockM[l,q] = NewBlockM[q,l]
	return NewBlock

def GaussianLikelihood (DM,mean,cov):
	#This function is meant to compute p(x|z_i,theta), where theta is set of parameters
	#Inputs:
		#DM: nxp array of data
		#mean: n-length vector mean for cluster of interest
		#cov: pxp covariance matrix
	
	storeprob = np.zeros(DM.shape[0])
	for i in range(DM.shape[0]):
		draw = multivariate_normal.logpdf(DM[i,:],mean=mean,sigma=cov)
		if(np.isinf(draw)):
			storeprob[i] = 0.00000001
		else:
			storeprob[i] = draw
	
	return storeprob
	
def logprobaSBM(DM,ExpMat,alphaMat,commind):
	#calculates p(a_i | z_ic)
	#inputs:
		#DM is nxn adjacency matrix
		#Exp mat is nxk matrix of expectations
		#alphamat is kxk matrix of probability parameters 
		#commind is community index of interest
	#output is n-length array of probabilities
	
	#place to store log probabilities
	logprobvec = np.zeros(DM.shape[0])

	#soft community assignments
	#SoftCommAssn = np.argmax(ExpMat,axis=1)
	
	for i in range(DM.shape[0]):
		SpecificNeighbors = np.argwhere(DM[i,:]==1)
		NonNeighbors = np.argwhere(DM[i,:]==0)
		logprobNeighbor = 0
		logprobNonNeighbor = 0
		for c in range(ExpMat.shape[0]):
			if(length(SpecificNeighbors)==0):
				logprobNeighbor = 0
			else:
				for j in range(length(SpecificNeighbors)):
					logprobNeighbor += ExpMat[SpecificNeighbors[j],c]*np.log(alphaMat[commind,c])		
		for k in range(length(NonNeighbors)):
			logprobNonNeighbor += ExpMat[NonNeighbors[k],c]*np.log(1-alphaMat[commind,c])
	
		logprobvec[i] = logprobNeighbor+logprobNonNeighbor
	return logprobvec

def logsum(vec):
	m=np.max(vec)
	return np.log(np.sum(np.exp(vec-m)))+m
	

def ComputeExpectation(AttributeMat,StoreMu,StoreCov,Alpha,k,logPi,TauMat,Network):
	#this finction takes means, covariances, etc and computes probability of node to community assn

#inputs
	#AttributeMat is the nxp matrix of attributes
	#StoreMu is the list object of Mus
	#StoreCov is the list object of estimated covariances
	#Alpha is the propensity 
	#k is the number of expected communities
	#logPi is probability of being in each of the communities 
	#TauMat is the prior version of Tau

	AttributeLL = np.zeros((AttributeMat.shape[0],k))
	for j in range(AttributeLL.shape[1]):
		AttributeLL[:,j] = GaussianLikelihood(AttributeMat,StoreMu[j,:],StoreCov[j])
	
	zeroInds = which(AttributeLL==0)
	AttributeLL[zeroInds] = 0.00000001
	
	###Now do the analog for graph #####
	GraphLL = np.zeros(AttributeLL.shape)
	for c in range(AttributeLL.shape[1]):
		GraphLL[:,c] = logprobaSBM(Network,TauMat,Alpha,c)+logPi[c]
	
	PreLogSum = AttributeLL+GraphLL
	LogSum = np.apply_along_axis(logsum, 1, PreLogSum)
	TauInit = np.exp(PreLogSum-LogSum)
	ZeroInds np.where(TauInit==0)
	TauInit[ZeroInds] = np.exp(-10)
	return TauInit

def MuUpdate (ExpMat,DataMat,commind):
	#this function performs update of mu for mixture of gaussian
	#assumes we have n observations and p dimensions, #k clusters
	#ExpMat: nxk matrix of probabilities 
		#ExpMat{ik} gives probability node i is in comm k
	#DataMat: Nxp matrix of data observations
	#commind: the cluster we are computing this for
	NewMu = np.zeros(DataMat.shape[1])
	Bottom = 0
	for i in range(DataMat.shape[0]):
		NewMu += ExpMat[i,commind]*DataMat[i,:]
		Bottom += ExpMat[i,commind]
	return NewMu/Bottom

def CovMatUpdate(Mu,ExpMat,DataMat,commind):
	#updates covariance matrix for mog
	#Assumes n observeations, p features, k clusters
	#Inputs:
		#Mu: the p-length vector of means
		#ExpMat: nxk matrix of expectations
		#DataMat: nxp matrix of data
		#commind: the community we are interested in updating 

	CovMat = np.zeros((DataMat.shape[1],DataMat.shape[1]))
	Bottom = 0
	for i in range(DataMat.shape[0])):
		#Extract row of DataMat and turn into column vector
		RelFeat = DataMat[i,:]
		DiffVec = RelFeat-Mu
		CovMat += ExpMat[i,commind]*np.outer(DiffVec,DiffVec)
		Bottom += ExpMat[i,commind]
	return CovMat/Bottom
		
def ObjectiveFunction(AttributeMat,StoreMean,StoreCov,Alpha,piMat,Network,ExpMat):
	#computes objective based on current exp mat, and mv gaussian parameters
	#Inputs:
		#AttributeMat: the n x p matrix of attributes
		#StoreMean: the k length list of means
		#StoreCov: the k length list of covariances
		#Alpha: the kxk propensity matrix
		#piMat: the vector of log probabilities of cluster assignments
		#Network: adjacency matrix
		#ExpMat: the current matrix of expectationd

	obj = 0

	#First compute attribute LL
	AttributeLL = np.zeros((AttributeMat.shape[0],len(piMat)))
	for j in range(AttributeLL.shape[1]):
		AttributeLL[:,j] = GaussianLikelihood(AttributeMat,StoreMean[j,:],StoreCov[j])
	
	zeroInds=np.where(AttributeLL==0)
	AttributeLL[zeroInds]<-0.0000001
	#AttributeLL<-log(AttributeLL)
	
	#Now compute graph LL
	GraphLL = np.zeros((AttributeMat.shape[0],len(piMat)))
	for c in range(AttributeLL.shape[1]):
		thing = logprobaSBM(Network,ExpMat,Alpha,c)
		GraphLL[:,c] = thing
		

	for i range(AttributeMat.shape[0]):
		for kc in range(len(piMat)):
			obj += (ExpMat[i,kc]*AttributeLL[i,kc])+(ExpMat[i,kc]*GraphLL[i,kc])+(ExpMat[i,kc]*piMat[kc])
			
	return obj

#################################################################################################################################

def fit(G,AttributeMat):
	communities = nx_comm.louvain_communities(G)
	k = len(communities)
	N = AttributeMat.shape[0]
	#create the feature mean and covariance measures
	StoreMean = []
	StoreCov = []
	##compute means and covariance matrices for each community###
	for kk in range(k):
		Inds = communities[kk]
		if(length(Inds)==0):
			StoreMean[kk] = np.zeros(AttributeMat.shape[1])
			StoreCov[k] = np.zeros((AttributeMat.shape[1],AttributeMat.shape[1]))
		else:
			StoreMean[kk] = np.mean(AttributeMat[Inds,],axis=0)
			StoreCov[kk] = np.cov(AttributeMat[Inds,])
			if(sum(StoreCov[kk])==0):
				StoreCov[kk] += 0.000001
			
	##compute the initial error##
	#compute an initial SBM connectivity profile 
	ProbMat = np.ones((N,k))*0.1
	for kk in range(k):
		for v in communities[kk]:
			ProbMat[v,communities[kk]] = 0.99
	Alpha=UpdateBlockParameters(ProbMat,Network)
	#create pivector or the probability of being in each of the class (k length vector)
	PiMat = np.mean(ProbMat,axis=0)
	logPi =  np.log(PiMat)
	#the first expectation is just the probability matrix
	Expect=ProbMat
	InitialObjective = ObjectiveFunction(AttributeMat,StoreMean,StoreCov,Alpha,logPi,Network,Expect)
	print(InitialObjective)
	##########################
	#begin iterative process
	##########################
	proceed=1
	Diff=2
	Objective=InitialObjective
	objectivevec = []
	while(proceed>0 & Diff>1):
	#update Mu and Cov
		for m in range(k):
			StoreMean[m] = MuUpdate(Expect,AttributeMat,m)
			StoreCov[m] = CovMatUpdate(StoreMean[m],Expect,AttributeMat,m)
			
		#compute new expectations based on these updated parameters 
		ExpectNew = ComputeExpectation(AttributeMat,StoreMean,StoreCov,Alpha,k,logPi,Expect,Network)

		#update the pi matrices of belonging to communities
		PiMat = np.mean(ExpectNew,axis=0)
		zeroInds = np.where(PiMat==0)
		PiMat[zeroInds] = 0.000001
		logPi = np.log(PiMat)

		#update SBM probability matrices
		AlphaNew = UpdateBlockParameters(ExpectNew,Network)

		#update the new objective value
		NewObjective = ObjectiveFunction(AttributeMat,StoreMean,StoreCov,AlphaNew,logPi,Network,ExpectNew)

		if(NewObjective>Objective):
			proceed=1
			##update everything##
			Alpha=AlphaNew
			Expect = ExpectNew
			Diff = np.abs(Objective-NewObjective)
			Objective=NewObjective
			objectivevec<-np.hstack([objectivevec,NewObjective])
			
		else:
			proceed=0
	
	return Expect 
