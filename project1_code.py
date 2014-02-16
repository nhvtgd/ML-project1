from string import punctuation, digits
import numpy as np
import matplotlib.pyplot as plt

def extract_words(input_string):
    """
      Returns a list of lowercase words in a strong.
      Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

def extract_dictionary(file):
    """
      Given a text file, returns a dictionary of unique words.
      Each line is passed into extract_words, and a list on unique
      words is maintained.
    """
    dict = []
    
    f = open(file, 'r')
    for line in f:
        flist = extract_words(line)
        
        for word in flist:
            if(word not in dict):
                dict.append(word)

    f.close()

    return dict

def extract_feature_vectors(file, dict):
    """
      Returns a bag-of-words representation of a text file, given a dictionary.
      The returned matrix is of shape (m, n), where the text file has m non-blank
      lines, and the dictionary has n entries. 
    """
    f = open(file, 'r')
    num_lines = 0

    for line in f:
        if(line.strip()):
            num_lines = num_lines + 1

    f.close()

    feature_matrix = np.zeros([num_lines, len(dict)])

    f = open(file, 'r')
    pos = 0
    
    for line in f:
        if(line.strip()):
            flist = extract_words(line)
            for word in flist:
                if(word in dict):
                    feature_matrix[pos, dict.index(word)] = 1
            pos = pos + 1
            
    f.close()
    
    return feature_matrix

def averager(feature_matrix, labels):
    """
      Implements a very simple classifier that averages the feature vectors multiplied by the labels.
      Inputs are an (m, n) matrix (m data points and n features) and a length m label vector. 
      Returns a length-n theta vector (theta_0 is 0). 
    """
    (nsamples, nfeatures) = feature_matrix.shape
    theta_vector = np.zeros([nfeatures])
    theta_zero = 0
    for i in xrange(0, nsamples):
        label = labels[i]
        sample_vector = feature_matrix[i, :]
        theta_vector = theta_vector + label*sample_vector
        #theta_zero += label
    return theta_vector, theta_zero

def read_vector_file(fname):
    """
      Reads and returns a vector from a file. 
    """
    return np.genfromtxt(fname)

def perceptron_algorithm(feature_matrix,labels):
    (nsamples,nfeatures) = feature_matrix.shape
    theta_vector = np.zeros([nfeatures])
    theta_zero = 0
    for i in range(20):
        mistake = False
        for i in xrange(0, nsamples):
            label = labels[i]
            sample_vector = feature_matrix[i,:]
            inner_prod = np.dot(theta_vector,sample_vector)
            sign = np.sign(label*(inner_prod + theta_zero))
            if sign <= 0:
                theta_vector += label*sample_vector
                theta_zero += label
                mistake = True
        if mistake == False: break
    return theta_vector, theta_zero
      
def passive_aggressive(feature_matrix,labels):
    (nsamples,nfeatures) = feature_matrix.shape
    theta_vector = np.zeros([nfeatures])
    theta_zero = 0
    for counter in range(10):
        for i in xrange(0, nsamples):
            label = labels[i]
            sample_vector = feature_matrix[i,:]
            inner_prod = np.dot(theta_vector,sample_vector)
            loss = max(0, 1 - label*inner_prod)
            eta = loss/np.linalg.norm(sample_vector)**2
            theta_vector += eta*label*sample_vector
    return theta_vector, theta_zero

def cross_validate(feature_matrix, labels, k, algorithm):
    (nsamples, nfeatures) = feature_matrix.shape
    numOfPartition = nsamples/k
    numberOfMistakes = []
    for i in range(numOfPartition):
        test_set = feature_matrix[i*k:(i+1)*k]
        labels_test_set = labels[i*k:(i+1)*k]
        training_set = __getValidConcatArray(feature_matrix[:i*k], feature_matrix[(i+1)*k:])
        labels_training_set = __getValidConcatArray(labels[:i*k],labels[(i+1)*k:])
        theta_vector, theta_zero = algorithm(training_set, labels_training_set)
        mistake = 0
        for index,test_data in enumerate(test_set):
            inner_prod = np.dot(test_data,theta_vector)
            if np.sign(labels_test_set[index]*(inner_prod + theta_zero)) <= 0:
                mistake += 1
        numberOfMistakes.append(mistake)
    return numberOfMistakes, len(numberOfMistakes)

def __getValidConcatArray(l1,l2):
    first_part = l1
    second_part = l2
    result = []
    if len(first_part) == 0:
        result = second_part
    elif len(second_part) == 0:
        result = first_part
    else:
        try:
            result = first_part + second_part
        except:
            result = np.array(first_part.tolist() + second_part.tolist())
    return result

def perceptron_classify(feature_matrix, theta_0, theta_vector):
    """
      Classifies a set of data points given a weight vector and offset.
      Inputs are an (m, n) matrix of input vectors (m data points and n features),
      a real number offset, and a length n parameter vector.
      Returns a length m label vector. 
    """
    (nsamples, nfeatures) = feature_matrix.shape
    label_output = np.zeros([nsamples])
    theta0 = theta_0
    for i in xrange(0, nsamples):
        sample_features = feature_matrix[i, :]
        perceptron_output = theta0 + np.dot(theta_vector, sample_features)
        
        if(perceptron_output > 0):
            label_output[i] = 1
        else:
            label_output[i] = -1

    return label_output

def write_label_answer(vec, outfile):
    """
      Outputs your label vector the a given file.
      The vector must be of shape (70, ) or (70, 1),
      i.e., 70 rows, or 70 rows and 1 column.
    """
    
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return

    for v in vec:
        if((v != -1.0) and (v != 1.0)):
            print("Invalid value in input vector.")
            print("Aborting write.")
            return
        
    np.savetxt(outfile, vec)
        

def plot_2d_examples(feature_matrix, labels, theta_0, theta):
    """
      Uses Matplotlib to plot a set of labeled instances, and
      a decision boundary line.
      Inputs: an (m, 2) feature_matrix (m data points each with
      2 features), a length-m label vector, and hyper-plane
      parameters theta_0 and length-2 vector theta. 
    """
    
    cols = []
    xs = []
    ys = []
    
    for i in xrange(0, len(labels)):
        if(labels[i] == 1):
            cols.append('b')
        else:
            cols.append('r')
        xs.append(feature_matrix[i][0])
        ys.append(feature_matrix[i][1])

    #plt.scatter(xs, ys, s=40, c=cols)

    [xmin, xmax, ymin, ymax] = [min(xs),max(xs),min(ys),max(ys)]

    linex = []
    liney = []
    for x in np.linspace(xmin, xmax):
        linex.append(x)
        if(theta[1] != 0.0):
            y = (-theta_0 - theta[0]*x) / (theta[1])
            liney.append(y)
        else:
            liney.append(0)
    
    #plt.plot(linex, liney, 'k-')

    #plt.show()
    return xs,ys,cols, linex, liney
