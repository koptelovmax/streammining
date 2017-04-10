import numpy as np

NMIN = 200 # grace period (minimum number of examples for learning one attribute)
DELTA = 0.01#0.0000001 # 0.99#1e-7 # split confidence: constant = 1 - probability that correct attribute is chosen
TAU = 0.05 # tie-breaking parameter
#%%

class Hleaf():
    def __init__(self,numClasses=2,posAttrib=[]):
        self.split = None # instance of HNode class
        self.nl = 0 # number of examples seen in a leaf since last split
        self.n = 0 # real counter
        self.stat = np.zeros(numClasses,int) # number of examples w.r.t. classes (Pos,Neg,... R in total)
        self.candidates = []
        for attribute in posAttrib:
            self.candidates.append(HNode(attribute.getAttributeId(),attribute.getNumberValues(),numClasses))

    def getStat(self):
        return self.stat

    def setStat(self,newStat):
        self.stat = newStat[:]
        
    def getCounts(self):
        return self.nl

    def setCounts(self,newCounts):
        self.nl = newCounts
        
    def getNumClasses(self):
        return len(self.stat)

    def checkHBound(self,G1,G2,text_file):
        R = np.round(np.log2(len(self.stat))) # range of bound
        epsilon = np.sqrt((R*R*np.log(1/np.float(DELTA)))/np.float(2*self.nl))
        #print 'Example',self.nl,'G1 =',G1,'G2 =',G2,'bound =',epsilon
        dG = np.abs(G1 - G2)
        if (dG > epsilon) or (epsilon < TAU):
            ##print 'HBound is satisfied, number of examples',self.n,np.sum(self.stat),self.nl,G1,G2,epsilon,(dG > epsilon),(epsilon < TAU)
            ##text_file.write('HBound is satisfied, number of examples'+str(self.nl)+'\n')
            return True
        else:
            return False
            
    def predictClass(self):
        return list(self.stat).index(np.max(self.stat))

    def countNumberOfLeaves(self):
        if self.split == None:
            ##return 1
            if np.sum(self.stat) != 0:
                return 1
            else:
                return 0
        else:
            return self.split.countNumberOfLeaves()
        
    def countNumberOfNodes(self):
        if self.split == None:
            ##return 1
            if np.sum(self.stat) != 0:
                return 1
            else:
                return 0
        else:
            return self.split.countNumberOfNodes()
            
    def printTree(self,string):
        if self.split == None:
            ##print string,'Leaf',self.stat,np.sum(self.stat),self.n,self.nl
            if np.sum(self.stat) != 0:
                print string,'Leaf',self.stat
        else:
            self.split.printTree(string)
            
    def countDepth(self):
        if self.split == None:
            return 0
        else:
            return self.split.countDepth()

    def sortExampleAndPredict(self,example,text_file):
        classIdPredicted = None
        classId = example[-1]
        if self.split == None:
            classIdPredicted = self.predictClass() # predict class by majority selection
            ##text_file.write(str(self.nl)+'\n')            
            self.nl += 1
            self.n += 1
            self.stat[classId] += 1
            if (self.candidates != []):
                for attribute in self.candidates:
                    attribute.sortExample(example,text_file)
                #if ((self.nl > NMIN) and (self.checkSameClass() != True)):
                ##if ((np.mod(self.nl,NMIN) == 0) and (self.checkSameClass() != True)):
                ##if ((np.mod(np.sum(self.stat),NMIN) == 0) and (self.checkSameClass() != True)):
                if ((np.mod(self.n,NMIN) == 0) and (self.checkSameClass() != True)):
                    ####print "Stat:",[self.candidates[i].getStat() for i in range(len(self.candidates))]
                    ##gainList = [self.candidates[i].Gini() for i in range(len(self.candidates))]
                    if len(self.candidates) == 1:
                        self.splitLeaf(self.candidates[0])
                    else:
                        gainList = [self.candidates[i].Gain(self.stat) for i in range(len(self.candidates))]
                        ##print 'gainList',gainList
                        ##gainList.append(0) # consider also null attribute (it has Gain=0, because it doesn't split)
                        bestGain1 = np.max(gainList)
                        gainList2 = gainList[:]      
                        gainList2.remove(bestGain1)
                        bestGain2 = np.max(gainList2)
                        if self.checkHBound(bestGain1,bestGain2,text_file):
                            #if np.abs(bestGain1) > np.abs(bestGain2):
                            if bestGain1 > bestGain2:
                                ##print 'Split suggestion',bestGain1
                                self.splitLeaf(self.candidates[gainList.index(bestGain1)])
                            else:
                                ##print 'Split suggestion',bestGain2
                                self.splitLeaf(self.candidates[gainList.index(bestGain2)])
        else:
            classIdPredicted = self.split.sortExampleAndPredict(example,text_file) # sort example into following attribute
        return classIdPredicted
        
    def sortExample(self,example,text_file):
        classId = example[-1]
        #text_file.write(str(self.nl)+'\n')        
        self.nl += 1
        self.n += 1
        self.stat[classId] += 1
                     
    def splitLeaf(self,attribToSplit):
        attributeId = attribToSplit.getAttributeId()
        numValues = attribToSplit.getNumberValues()
        numClasses = len(self.stat)
        self.candidates.remove(attribToSplit)
        ##print 'Split leaf by attribute',attributeId,'next candidates',[i.getAttributeId() for i in self.candidates]
        self.split = HNode(attributeId,numValues,numClasses,self.candidates[:])
        candStat = attribToSplit.getStat()
        self.split.setStat(candStat)
        #print 'CandStat',candStat#,'Gain',self.split.Gain(self.stat)
        ##print 'newStat',self.split.getStat()
        candCounts = attribToSplit.getCounts()
        ##print 'CandCounts',candCounts
        self.split.setCounts(candCounts)
        ##print 'newCounts',self.split.getCounts()
        #attribToSplit.resetStat()         
        ##self.split = attribToSplit #HNode(attributeId,numValues,numClasses,self.candidates[:])
        self.candidates = []
        self.nl = 0
        self.n = 0
        self.stat = np.zeros(numClasses,int)
        
    def checkSameClass(self):
        if np.max(self.stat[:]) == np.sum(self.stat):
            return True
        else:
            return False
    
    def Entropy(self):
        num = np.sum(self.stat)
        if num != 0:
            result = 0
            for i in range(len(self.stat)):
                ratio = self.stat[i]/np.float(num)
                if ratio != 0:
                    result += ratio*np.log2(ratio)
            return -result
        else:
            return 0

    def Gini(self):
        num = np.sum(self.stat)
        if num != 0:
            result = 0
            for i in range(len(self.stat)):
                ratio = self.stat[i]/np.float(num)
                result += ratio*ratio
            return 1-result
        else:
            return 0
        

class HNode():

    def __init__(self,attributeId,numValues=2,numClasses=2,posAttrib=[]):
        self.attr_id = attributeId # attribute id node represents
        self.values = [] # list of leaves (attribute values)
        for i in range(numValues):
            self.values.append(Hleaf(numClasses,posAttrib[:]))
            
    def getAttributeId(self):
        return self.attr_id
        
    def getNumberValues(self):
        return len(self.values)
        
    def getStat(self):
        result = []
        for value in self.values:
            result.append(value.getStat())
        return result
        
    def getCounts(self):
        result = []
        for value in self.values:
            result.append(value.getCounts())
        return result

    def setStat(self,newStat):
        for i in range(len(self.values)):
            self.values[i].setStat(newStat[i])
            
    def setCounts(self,newCounts):
        for i in range(len(self.values)):
            self.values[i].setCounts(newCounts[i])
                                
    def countNumberOfLeaves(self):
        numLeaves = 0
        for value in self.values:
            numLeaves += value.countNumberOfLeaves()
        return numLeaves
        
    def countNumberOfNodes(self):
        numNodes = 1
        for value in self.values:
            numNodes += value.countNumberOfNodes()
        return numNodes
        
    def printTree(self,string):
        print string,'Attribute',self.attr_id
        for value in self.values:
            value.printTree(string+'     ')
        
    def countDepth(self):
        maxDepth = 0
        for value in self.values:
            depth = value.countDepth()
            if depth > maxDepth:
                maxDepth = depth
        return 1+maxDepth

    def sortExample(self,example,text_file):
        attribValue = example[self.attr_id]
        self.values[attribValue].sortExample(example,text_file)

    def sortExampleAndPredict(self,example,text_file):
        attribValue = example[self.attr_id]       
        return self.values[attribValue].sortExampleAndPredict(example,text_file)
        
    def Gain(self,preSplitDist):
        n = np.sum(preSplitDist)
        if n != 0:
            # Entropy (preSplitDist):
            result = 0
            for i in range(len(preSplitDist)):
                ratio = preSplitDist[i]/np.float(n)
                if ratio != 0:
                    result += ratio*np.log2(ratio)
            preSplitEntropy = -result
            # Entropy (postSplitDist):
            n = 0
            result = 0         
            for leaf in self.values:
                ratio = np.sum(leaf.getStat())
                n += np.sum(leaf.getStat())
                result += ratio*leaf.Entropy()
            postSplitEntropy = result/np.float(n)
            return preSplitEntropy-postSplitEntropy
        else:
            return 0

    def Gini(self):
        numClasses = self.values[0].getNumClasses()
        nodeStat = np.zeros(numClasses,int)
        for leaf in self.values:
            nodeStat += np.array(leaf.getStat())
        n = np.sum(nodeStat)
        if n != 0:
            # Gini:
            result = 0
            for i in range(numClasses):
                ratio = nodeStat[i]/np.float(n)
                result += ratio*ratio
            gini = 1-result
            # Gain:
            result = 0         
            for leaf in self.values:
                ratio = np.sum(leaf.getStat())/np.float(n)
                result += ratio*leaf.Gini()
            return gini-result
        else:
            return 0

##            
def Entr(val):
    result = 0
    for i in range(len(val)):
        ratio = val[i]/np.float(np.sum(val))
        if ratio != 0:
            result += ratio*np.log2(ratio)
    return -result

def Gi(val):
    result = 0
    for i in range(len(val)):
        ratio = val[i]/np.float(np.sum(val))
        result += ratio*ratio
    return 1-result

def Ga(val1,val2):
    ratio1 = np.sum(val1)/np.float(np.sum(val1+val2))
    ratio2 = np.sum(val2)/np.float(np.sum(val1+val2))
    result = Entr([val1[0]+val2[0],val1[1]+val2[1]]) - (ratio1*Entr(val1) + ratio2*Entr(val2))
    return result

def Giga(val1,val2):
    ratio1 = np.sum(val1)/np.float(np.sum(val1+val2))
    ratio2 = np.sum(val2)/np.float(np.sum(val1+val2))
    result = Gi([val1[0]+val2[0],val1[1]+val2[1]]) - (ratio1*Gi(val1) + ratio2*Gi(val2))
    return result

def HBound(R,d,n):
    return np.sqrt((R*R*np.log(1/np.float(d)))/np.float(2*n))
#%%
# Initial values:
##c = 2 # number of classes
c = 10 # number of classes
#givenAttrib = [5,2,2,2,2,5,5,5] # dimensionality w.r.t. every attribute
##givenAttrib = [2,2] # dimensionality w.r.t. every attribute
givenAttrib = [4,12,4,13,4,13,4,13,4,12] # dimensionality w.r.t. every attribute

# Initialization of parameters:
candAttrib = []
for i in range(len(givenAttrib)):
    candAttrib.append(HNode(i,givenAttrib[i],c))
    
HT = Hleaf(c,candAttrib) # create a root leaf
count = 0 # number of examples
countCorrect = 0 # number of correctly classified examples
with open("results.txt", "w") as text_file:
    #f = open('/storage/sdcard0/org.qpython.qpy/scripts/poker.csv', 'r')
    #f = open('/storage/sdcard0/org.qpython.qpy/scripts/test.txt', 'r')
    ##f = open('stream.txt', 'r')
    ##f = open('test.txt', 'r')
    f = open('poker2.csv', 'r')
    for line in f:
        if count < 1000000:
            try:
                data = line.split(',')
                count += 1
                try:
                    #example = [np.int(data[0]),np.int(data[1]),np.int(data[2])]
                    example = [np.int(i) for i in data]
                    classPredicted = HT.sortExampleAndPredict(example,text_file)
                    if example[-1] == classPredicted:
                        countCorrect += 1
                    if np.mod(count,100000)==0:
                        print 'Iteration',count,'nodes',HT.countNumberOfNodes(),'leaves',HT.countNumberOfLeaves(),'depth',HT.countDepth()
                except ValueError:
                    print "Invalid example:", example, count
            except ValueError:
                print "Invalid input:", line
    f.close()
print '\nTotal examples:',count,'Correctly classified:',countCorrect,'Accuracy:', 100*countCorrect/float(count)
print '\nNumber of nodes (attributes+leaves):',HT.countNumberOfNodes()
print 'Number of leaves:',HT.countNumberOfLeaves()
print 'Tree depth (levels of attributes):',HT.countDepth()
print '\nPrint tree:'
HT.printTree('')
#%%
