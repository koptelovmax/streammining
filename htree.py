import numpy as np

class Hleaf():

    def __init__(self,numClasses=2):
      self.attr_id = None # attribute id leaf belongs to (None = Root HT)
      self.attr_val = None # attribute value leaf represents (None = Root HT)
      self.n_class = numClasses
      self.n = 0 # number of examples seen in leaf
      self.stat = np.zeros(numClasses,int) # statistics
      self.val = None # value for prediction
    
    def setLeafId(self,attribute,value):
        self.attr_id = attribute
        self.attr_val = value

    def getLeafId(self):
        return self.attr_id,self.attr_val
    
    def getStat(self):
        return self.stat

    def getExamplesSeen(self):
        return self.n
        
    def sortExample(self,classId):
        self.stat[classId] += 1
        self.n += 1
        
    def checkSameClass(self):
        if np.min(self.stat[:]) == 0:
            return True
        else:
            return False

    def Entropy(self):
        result = 0
        for i in range(self.n_class):
            ratio = self.stat[i]/np.float(np.sum(self.stat[:]))
            if ratio != 0:
                result += ratio*np.log2(ratio)
        return -result
        
#%%
class HNode():

    def __init__(self,numClasses=2):
      self.attr_id = None # attribute id leaf belongs to
      self.left = None
      self.right = None
      self.n_class = numClasses
      self.stat = np.zeros(numClasses,int) # statistics
      #self.stat = np.zeros((n_values,n_classes),int) # sufficient statistics
      self.n = 0 # number of examples seen

    def getNodeId(self):
        return self.attr_id
    
    def setNodeId(self,attribute):
        self.attr_id = attribute

    def setLeftChild(self,leaf,newNode):
        if leaf == True:
            self.left = Hleaf(newNode)
        else:
            self.left = newNode

    def getLeftChild(self):
        return self.left
        
    def setRightChild(self,leaf,newNode):
        if leaf == True:
            self.right = Hleaf(newNode)
        else:
            self.right = newNode

    def getRightChild(self):
        return self.right

    def getExamplesSeen(self):
        return self.n

    def sortExample(self,attribValue,classId):
        self.stat[classId] += 1
        self.n += 1
        if attribValue == 0:
            if self.left == None:
                newLeaf = Hleaf()
                #newLeaf.setLeafId(attribId,attribValue)
                newLeaf.sortExample(classId)
                self.left = newLeaf
            else:
                self.left.sortExample(classId)
        elif attribValue == 1:
            if self.right == None:
                newLeaf = Hleaf()
                #newLeaf.setLeafId(attribId,attribValue)
                newLeaf.sortExample(classId)
                self.right = newLeaf
            else:
                self.right.sortExample(classId)
                
    def checkSameClass(self):
        if np.min(self.stat[:]) == 0:
            return True
        else:
            return False
                
    def resetStat(self):
        self.n = 0
        self.stat = np.zeros(self.n_class,int)
        
    def Entropy(self):
        result = 0
        for i in range(self.n_class):
            ratio = self.stat[i]/np.float(np.sum(self.stat[:]))
            if ratio != 0:
                result += ratio*np.log2(ratio)
        return -result
    
    def childExamples(self,childId):
        if childId == 0:
            if self.left != None:
                return np.sum(self.left.getStat())
            else:
                return 0
        elif childId == 1:
            if self.right != None:
                return np.sum(self.right.getStat())
            else:
                return 0

    def childEntropy(self,childId):
        if childId == 0:
            if self.left != None:
                return self.left.Entropy()
            else:
                return 0
        elif childId == 1:
            if self.right != None:
                return self.right.Entropy()
            else:
                return 0
    
    def Gain(self):
        result = 0
        for i in range(self.n_class):
            ratio = self.childExamples(i)/np.float(np.sum(self.stat[:]))
            result += ratio*self.childEntropy(i)
        return self.Entropy()-result

#%%
def Entr(val):
    result = 0
    for i in range(len(val)):
        ratio = val[i]/np.float(np.sum(val))
        if ratio != 0:
            result += ratio*np.log2(ratio)
    return -result
    
def Ga(val1,val2):
    ratio1 = np.sum(val1)/np.float(np.sum(val1+val2))
    ratio2 = np.sum(val2)/np.float(np.sum(val1+val2))
    result = Entr([val1[0]+val2[0],val1[1]+val2[1]]) - (ratio1*Entr(val1) + ratio2*Entr(val2))
    return result

def HBound(R,d,n):
    return np.sqrt((R*R*np.log(1/np.float(d)))/np.float(2*n))

#%%
R = 2 # number of classes
delta = 1e-07 # constant = 1 - probability that correct attribute is chosen
nmin = 100 # minimum number of examples for learning one attribute

HT = Hleaf() # create a root leaf
A1 = HNode() # attribute 1
A2 = HNode() # attribute 2

level = 0

f = open('stream.txt', 'r')
for line in f:
    try:
        data = line.split(',')
        try:
            if level == 0:
                HT.sortExample(np.int(data[2]))
                A1.sortExample(np.int(data[0]),np.int(data[2]))
                A2.sortExample(np.int(data[1]),np.int(data[2]))
                n = HT.getExamplesSeen() # examples seen so far
                
                if (np.mod(n,nmin) == 0) and (HT.checkSameClass() != True):
                    G1 = A1.Gain()
                    G2 = A2.Gain()
                    epsilon = HBound(R,delta,n)
                    print n,G1,G2,epsilon
                    if np.abs(G1 - G2) > epsilon:
                        print n#,G1,G2,epsilon
                        if G1 >= G2:
                            HT = A1
                            A21 = HNode() # attribute 2 for left branch of attribute 1
                            A22 = HNode() # attribute 2 for right branch of attribute 1
                        else:
                            HT = A2
                            A11 = HNode() # attribute 1 for left branch of attribute 2
                            A12 = HNode() # attribute 1 for right branch of attribute 2
                        HT.resetStat()
                        level += 1
                        
            elif level == 1:
                if G1 >= G2:
                    HT.sortExample(np.int(data[0]),np.int(data[2]))
                    if np.int(data[0]) == 0:
                        A21.sortExample(np.int(data[1]),np.int(data[2]))
                    elif np.int(data[0]) == 1:
                        A22.sortExample(np.int(data[1]),np.int(data[2]))
                else:
                    HT.sortExample(np.int(data[1]),np.int(data[2]))
                    if np.int(data[1]) == 0:
                        A11.sortExample(np.int(data[0]),np.int(data[2]))
                    elif np.int(data[1]) == 1:
                        A12.sortExample(np.int(data[0]),np.int(data[2]))
                    
                n = HT.getExamplesSeen() # examples seen so far
                if (np.mod(n,nmin) == 0) and (HT.checkSameClass() != True):
                    if G1 >= G2:
                        G21 = A21.Gain()
                        G22 = A22.Gain()
                        epsilon = HBound(R,delta,n)
                        print n,G21,G22,epsilon
                        if np.abs(G21 - G22) > epsilon:
                            print n
                            if G21 >= G22:
                                HT.setLeftChild(False,A21)
                                print 'case 1'
                            else:
                                HT.setRightChild(False,A22)
                                print 'case 2'
                            HT.resetStat()
                            level += 1                
                    else:
                        G11 = A11.Gain()
                        G12 = A12.Gain()
                        epsilon = HBound(R,delta,n)
                        print G11,G12,epsilon
                        if np.abs(G11 - G12) > epsilon:
                            print n
                            if G11 >= G12:
                                HT.setLeftChild(False,A11)
                                print 'case 3'
                            else:
                                HT.setRightChild(False,A12)
                                print 'case 4'
                            HT.resetStat()
                            level += 1

        except ValueError:
            print "Invalid input:", data
    except ValueError:
        print "Invalid input:", line
f.close()

#print HT.checkSameClass()
#print HT.Entropy()
#print A1.Gain()
#print A2.Gain()

#%%
