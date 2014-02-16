import numpy
def Perceptron(inputDict, dimension, startPoint, orderList,theta):
    print "Order list :{0}".format(orderList)
    thetaInit, theta0 = None, None
    if not theta :
        thetaInit = [0]*dimension
    else:
        thetaInit, theta0 = theta
    #orderList = getOrderList(startPoint, inputDict.keys())
    converge = 0
    bookKeeping = {}
    while converge != len(inputDict):
        converge = 0
        for counter,i in enumerate(orderList):
            test = inputDict[i]*(numpy.dot(thetaInit,i) + theta0)
            if numpy.sign(test) <= 0:
                converge = 0
                if i not in bookKeeping:
                    bookKeeping[i] = 1
                else:
                    bookKeeping[i] += 1
                thetaInit = numpy.matrix(thetaInit) + inputDict[i]*numpy.matrix(i)
                thetaInit = thetaInit.tolist()[0]
                if theta0 != None:
                    theta0 = theta0 + inputDict[i]
            else:
                converge += 1
            print thetaInit, theta0, inputDict[i], converge

    return thetaInit, theta0, bookKeeping, "Number Mistake {0}".format(sum([bookKeeping[i] for i in bookKeeping]))

def plotProgress(array):
    soa = numpy.array(array)
    pass


def getOrderList(startPoint, collection):
    result = [startPoint]
    for data in collection:
        if data not in result:
            result.append(data)
    return result

result = []
def permutation(array):
    global result
    if len(array) == 1:
        result.append(array)
        return result
    else:
        print array, "array"
        for i in range(len(array)):
            for perm in permutation(array[:i] + array[i+1:]):
                result.extend(perm + [array[i]])                   
        return result

class Permutation:
    def __init__(self, justalist):
        self._data = justalist[:]
        self._sofar = []
    def __iter__(self):
        return self.next()
    def next(self):
        for elem in self._data:
            if elem not in self._sofar:
                self._sofar.append(elem)
                if len(self._sofar) == len(self._data):
                    yield self._sofar[:]
                else:
                    for v in self.next():
                        yield v
                self._sofar.pop()

def testPerceptronAlg(theta, testData):
    thetaInit, theta0 = theta
    for data in testData:
        test = testData[data]*(numpy.dot(thetaInit,data) + theta0)
        if numpy.sign(test) <= 0:
            print "Test Error Point {0} label {1} theta {2}".format(data, testData[data], theta)
            
import random
inputDict = {tuple([2,3]):1, tuple([3,3]): 1, tuple([3,4]): 1, tuple([4,1]):1,tuple([5,1]):-1,tuple([3,0]):-1}
dimension = 2
inputDict2 = {tuple([-3,2]):1, tuple([-1,1]): 1, tuple([-1,-1]):-1,tuple([2,2]):-1, tuple([1,-1]) : -1}
#print Perceptron(inputDict2, dimension, tuple([-3,-1]), [tuple([-3,2]), tuple([-1,1]), tuple([-1,-1]), tuple([2,2]) ,tuple([1,-1])], theta = [[0,0],0])

x = {tuple([3+random.randint(1,10),3+random.randint(1,10)]):1 for i in range(5)}
y = {tuple([2-random.randint(1,10),2-random.randint(1,10)]):-1 for i in range(5)}
test = dict(x.items() + y.items())
#print Perceptron(inputDict2, dimension, tuple([-3,-1]), test.keys(),theta = [[0,0],0])
#testPerceptronAlg([[-3,6],-2], test)
#testPerceptronAlg([[-2,2],1], test)

x = {tuple([-2+i,1]):1 for i in range(5)}
y = {tuple([-2+i,-1]):-1 for i in range(5)}
test = dict(x.items() + y.items())
print Perceptron(test,2,None, test.keys(), theta = [[0,0],0])
print test
func = [x[1] for x in test.keys()]
func2 = [x[1] for x in test.items()]
a = dict(zip(func,func2))
print a
print Perceptron(a,1,None, a.keys(), theta = [[0],0])
