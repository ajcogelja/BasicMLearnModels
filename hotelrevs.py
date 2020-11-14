import pandas as pd
import numpy as np

def get_eigen(mat):
    return np.linalg.eig(mat)

def multiply(n, m):
    return np.matmul(n, m)

def dot_product(weights, words, bias):
    dot = bias
    index = 0
    for w in weights:
        dot += w*words[index]
        index += 1
    return dot

def within_margin(test_val, margin, expected_val):
    if test_val + margin >= expected_val and test_val - margin <= expected_val:
        return True
    else:
        return False

if __name__ == "__main__":
    data = pd.read_csv('tripadvisor_hotel_reviews.csv')
    data = data.sample(1000).reset_index()
    #convert data into term vector maybe?
    output_label = 'Rating' 
    text_label = 'Review'
    weights = []
    default_weight = 1
    
    #this is used to give the words dictionary a positional value, so weights correspond to specific words
    all_words = {}
    for entry in data[text_label]:
        for w in entry.split(' '):
            all_words[w] = True

    generic_word_vector = {}
    index = 0
    #Can try different models with different subsets of the words to see which are most relevant
    max_index = len(all_words)/50
    print('max_index:', max_index)
    for word in all_words:
        if index >= max_index:
            break
        generic_word_vector[word] = 0
        index += 1
    
    MAX_LENGTH = len(generic_word_vector)
    bias = default_weight
    #initializes the weight vector
    for x in range(MAX_LENGTH):
        col = []
        for y in range(MAX_LENGTH):
            col.append(default_weight)
        weights.append(col)
    #vec = [1, 2, 5]
    #vec = [1, 1, 1]
    #actual_vec = np.array([vec])
    #actual_vec.reshape(1, len(vec))
    #vec_trans = np.transpose(actual_vec)#actual_vec.reshape(len(vec), 1)
    #print('vec: ', actual_vec, '\n, vec_trans: ', vec_trans)
    #print('multiple: ', np.matmul(vec_trans, actual_vec))

    #training
    iteration = 0
    #row = 0
    learning_rate = .65
    max_iters = 1000

    word_vecs = []
    for entry in data[text_label]:
            word_vector = generic_word_vector
            size = 0
            #generate vector for entry
            for word in entry.split(' '):
                if word not in word_vector:
                    continue
                word_vector[word] += 1
                size += 1
            #normalize word vector
            for v in word_vector:
                word_vector[v] = word_vector[v]/size #converts the words into frequencies instead of counts. (maybe not a good idea. we will see)
            normal_vector = np.array([list(word_vector.values())])
            normal_trans = normal_vector.transpose()
            mat = multiply(normal_trans, normal_vector)
            word_vecs.append(mat)
    print('beginning training')
    while iteration < max_iters:
        correct_examples = 0
        total_examples = 0
        row = 0
        for entry in word_vecs:
            #dot = dot_product(weights, word_vector, bias) <- would be used if we wanted to do a perceptron or something with just these weights, but i want the 2d array!
            #word_vector.items() gets all the pairs
            #normal_vector = np.array([list(word_vector.values())])
            #normal_vector = np.array([list(entry.values())])
            #normal_trans = normal_vector.transpose()
            #print('normal vector: ', normal_vector, '\n')
            #print('normal vector: ', normal_vector, '\ntranspose: ', np.transpose(normal_vector))
            #square_mat = multiply(normal_trans, normal_vector)
            #print('shape: ', square_mat.shape)
            #h = square_mat.shape[0]
            #w = square_mat.shape[1]
            #square_mat.reshape(h*w, 1)
            square_mat = entry
            product =  bias#np.inner(square_mat, weights) + bias
            x_index = 0
            for w in weights:
                y_index = 0
                for e in w:
                    product += (e*square_mat[x_index, y_index])
                    y_index += 1
                x_index += 1

            output = data[output_label][row]
            total_examples += 1
            if not within_margin(product, .25, output):
                #incorrectly labelled examples
                diff = output - product #output = 4, product = 2, weights increase, 
                diff = diff * learning_rate
                bias += diff
                x_index = 0
                for w in weights:
                    y_index = 0
                    for e in w:
                        #product += (e*square_mat[x_index, y_index])
                        weights[x_index][y_index] += diff*square_mat[x_index, y_index]
                        y_index += 1
                    x_index += 1
            else:
                correct_examples += 1
                #print('predicted correctly: ', product, 'actual out: ', output)

            row += 1
        learning_rate = learning_rate * .99 #slowly decrease learning rate
        print('iteration: ', iteration, ' accuracy', correct_examples/total_examples)
        iteration += 1

        #now we dot square_mat with weights
    #print('data:\n', data[output_label])