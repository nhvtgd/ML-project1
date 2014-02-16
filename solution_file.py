import project1_code as p1
import time
import example_plot as plot
import numpy as np
import matplotlib.pyplot as plt

dictionary = p1.extract_dictionary('train-tweet.txt')
labels = p1.read_vector_file('train-answer.txt')
feature_matrix = p1.extract_feature_vectors('train-tweet.txt', dictionary)
def testAlgorithm(algorithmToTest , name):    
    start = time.clock()
    average_theta,theta_0 = algorithmToTest(feature_matrix, labels)
    stop = time.clock()
    label_output = p1.perceptron_classify(feature_matrix, theta_0, average_theta)
    
    correct = 0
    for i in xrange(0, len(label_output)):
        if(label_output[i] == labels[i]):
            correct = correct + 1

    percentage_correct = 100.0 * correct / len(label_output)
    print(name + " gets " + str(percentage_correct) + "% correct (" + str(correct) + " out of " + str(len(label_output)) + ").")
    return average_theta, theta_0, stop - start

def testPlot(algorithmFnc, dataFnc,figNum,figName):
    feature_matrix, labels = dataFnc(True)
    theta,theta0 = algorithmFnc(np.array(feature_matrix), labels)
    xs,ys,cols, linex, liney = p1.plot_2d_examples(feature_matrix, labels, theta0, theta)

    feature_matrix1, labels1 = dataFnc(False)
    theta1,theta01 = algorithmFnc(np.array(feature_matrix1), labels1)
    xs1,ys1,cols1, linex1, liney1 = p1.plot_2d_examples(feature_matrix1, labels1, theta01, theta1)    

    plt.figure(figNum)
    plt.subplot(211)
    plt.scatter(xs,ys,s=40,c=cols)
    plt.plot(linex, liney, 'k-')
    plt.title(figName, fontsize=30)
    plt.subplot(212)
    plt.scatter(xs1,ys1,s=40,c=cols1)
    plt.plot(linex1, liney1, 'k-')
    plt.show()

def testValidation(algorithm, k):
    return p1.cross_validate(feature_matrix, labels, k,algorithm)

def problem2Solution():
    print testAlgorithm(p1.averager, "Averager")
    print testAlgorithm(p1.perceptron_algorithm, "Perceptron")
    print testAlgorithm(p1.passive_aggressive, "Passive Aggressive")

def problem3Solution():
    testPlot(p1.averager,plot.synthetic_data_averager,1, "Averager")
    testPlot(p1.perceptron_algorithm, plot.synthetic_data_perceptron,2, "Perceptron")
    testPlot(p1.passive_aggressive, plot.synthetic_data_perceptron,3, "Passive Aggressive")
    
def problem4Solution():
    print testValidation(p1.passive_aggressive,10)
    print testValidation(p1.perceptron_algorithm,10)
    
problem2Solution()
problem4Solution()
    
