import os
import numpy as np
import random
import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class TrainDigits:
    def __init__(self, filename):
        self.filename = filename


    def read_images(filename, width, height, n_images):
        images = []
        if(os.path.exists(filename)):
            file=[l[:-1] for l in open(filename).readlines()]
            file.reverse()
            for i in range(n_images):
                image=[]
                for j in range(height):
                    image.append(list(file.pop()))
                if len(image[0]) < width-1:
                    # we encountered end of file...
                    print("Truncating at %d examples (maximum)" % i)
                    break
                images.append(image)
        return images

    def read_labels(filename, n_labels):
        labels=[]
        if(os.path.exists(filename)):
            file=[l[:-1] for l in open(filename).readlines()]
            for line in file[:min(n_labels, len(file))]:
                if line == '':
                    break
                labels.append(int(line))
        return labels

    def print_image(image):
        for row in image:
            print(''.join(row))

    # cuts image into 15 sections (14x20)
    # "each square in the grid defines a binary feature that indicates whether there is anything marked inside the square"
    def image_to_feature(image):
        #print("NEW IMAGE")
        # subgrids = []
        # rows = 28
        # cols = 28
        # subgrid_rows = 4
        # subgrid_cols = 4

        # result_array = [0] * 49
        # count = 0
        # for i in range(0, rows, subgrid_rows):
        #     for j in range(0, cols, subgrid_cols):
        #         for i_increase in range(4):
        #             for j_increase in range(4):
        #                 if image[i+i_increase][j+j_increase] == "+":
        #                     result_array[count] = 1
        #                 if image[i+i_increase][j+j_increase] == "#":
        #                     result_array[count] = 2

        #         count+=1
        count = 0
        result_array = [0] * 784
        for i in range(28):
            for j in range(28):
                if image[i][j] == "#":
                    result_array[count] = 1
                count+=1


        return result_array

    def initalize_weights():
        theta_1 = np.random.uniform(-1, 1, size=(784, 785))
        theta_2 = np.random.uniform(-1, 1, size=(10, 785))

        for i in range(784):
            for j in range(785):
                while(theta_1[i][j] == 0):
                    theta_1[i][j] = random.uniform(-1, 1)
        
        for k in range(10):
            for m in range(785):
                while (theta_2[k][m] == 0):
                    theta_2[k][m] = random.uniform(-1, 1)

        return theta_1, theta_2
    
    def forward_pass(theta_1, theta_2, image):
        if(len(image) != 784):
            print("ERROR")
            return
        image_numpy = np.array(image)
        bias_included = np.insert(image_numpy, 0, 1) #added bias
        x = bias_included.reshape(-1, 1)  #Reshape to (50, 1)
        z_2 = np.dot(theta_1, x)
        a_2_g = sigmoid(z_2)
        a_2 = np.insert(a_2_g, 0, 1, axis=0) #added bias of 1

        z_3 = np.dot(theta_2, a_2)
        a_3 = sigmoid(z_3)
        return x, a_3, a_2
    
    def compute_errors(a_3, a_2, label, theta1, theta2):
        y = np.zeros((10,1))
        for i in range(len(y)):
            if i == label:
                y[i] = 1

        transpose_theta1 = np.transpose(theta1)
        subset = theta2[:, 1:] # is 10 x 196
        transpose_theta2 = np.transpose(subset) # is 196 x 10
        delta_3 = a_3 - y # 10x1
        #first_half = np.dot(transpose_theta1, delta_3)

        #not including bias
        first_half = np.dot(transpose_theta2, delta_3)
        one_minus_a2 = 1 - a_2[1:]
        second_half = a_2[1:] * one_minus_a2 
        reshape_first_half = first_half.reshape((784, 1))
        delta_2 = reshape_first_half * second_half

        # print("a_3 : " + str(a_3.shape))
        # print("a_2 : " + str(a_2.shape))
        # print("theta1 : " + str(theta1.shape))
        # print("theta2 : " + str(theta2.shape))
        # print("delta_2 : " + str(delta_2.shape))
        # print("delta_3 : " + str(delta_3.shape))
        # print("transpose theta2 : " + str(transpose_theta2.shape))
        # print("first half : " + str(first_half.shape))
        # print("reshape half : " + str(reshape_first_half.shape))
        # print("second half : " + str(second_half.shape))
        # print("subset : " + str(subset.shape))
        # print(y)

        return delta_2, delta_3

    def compute_gradient(x, a_3, a_2, delta_2, delta_3, theta1_delta, theta2_delta):

        local_theta2_delta = np.dot(delta_3, np.transpose(a_2))
        local_theta1_delta = np.dot(delta_2, np.transpose(x))

        theta1_delta += local_theta1_delta

        theta2_delta = np.reshape(theta2_delta, (10, 785))
        theta2_delta += local_theta2_delta

        return theta1_delta, theta2_delta

    def regularize_gradient(theta1, theta2, theta1_delta, theta2_delta, iterations):
        lambda_reg = 0.22
        reg_theta1 = theta1_delta / iterations
        reg_theta2 = theta2_delta / iterations

        for val in range(reg_theta2.shape[0]):
            for ind in range(reg_theta2.shape[1]):
                if ind != 0: #ignore bias weight
                    reg_theta2[val][ind] += lambda_reg * theta2[val][ind]
        
        for i in range(reg_theta1.shape[0]):
            for j in range(reg_theta1.shape[1]):
                if j != 0: #ignore bias weight
                    reg_theta1[i][j] += lambda_reg * theta1[i][j]
        
        return reg_theta1, reg_theta2

    def loss(list_y, list_a_3):
        if len(list_y) != len(list_a_3):
            print("ERROR")
            return
        
        sum = 0
        for index in range(len(list_a_3)):
            a_3 = list_a_3[index]
            y = list_y[index]
            for i in range(10):
                first_term = y[i] * math.log2(a_3[i])
                second_term = (1 - y[i]) * math.log2(1 - a_3[i])
                sum += (first_term + second_term)
        
        return sum/(-1 * len(list_y))


    def get_random_sample(percent, training_images, training_labels):
        #sample size will be 0.1, 0.2, 1
        num_items_to_select = int (len (training_labels) * percent)
        indices = list(range(len (training_labels)) )
        random.shuffle(indices)
        selected_indices = indices [:num_items_to_select]
        selected_images = [training_images[i] for i in selected_indices]
        selected_labels = [training_labels[i] for i in selected_indices]
        return selected_images, selected_labels


    if __name__ == "__main__":
        n = 5000
        training_images=read_images("data/digitdata/trainingimages", 28, 28, n) # 28 rows with 28 columns
        training_labels=read_labels("data/digitdata/traininglabels", n)

        training_images_feature = []
        for image in training_images:
            training_images_feature.append(image_to_feature(image))
        
        percent = 0.09
        selected_images, selected_labels = get_random_sample(percent, training_images_feature, training_labels)


        theta1, theta2 = initalize_weights()
        for epoch in range(200):
            theta1_delta = np.zeros((784, 785))
            theta2_delta = np.zeros((10, 785))

            for image in range(len(selected_images)):
                #print(image)
                x, a_3, a_2 = forward_pass(theta1, theta2, selected_images[image])
                delta_2, delta_3 = compute_errors(a_3, a_2, selected_labels[image], theta1, theta2)
                theta1_delta, theta2_delta = compute_gradient(x, a_3, a_2, delta_2, delta_3, theta1_delta, theta2_delta)
        
            reg_theta1, reg_theta2 = regularize_gradient(theta1, theta2, theta1_delta, theta2_delta, len(selected_images))
            #update the weight gradients
            theta1 = theta1 - (0.019*reg_theta1) #0.015 was last best
            theta2 = theta2 - (0.019*reg_theta2) 

            print(epoch)
            # abs_arr = np.abs(reg_theta2)
            # average_abs = np.mean(abs_arr)

            # matches = 0
            # list_a_3 = []
            # list_y = []
            # if epoch %100 == 0: 
            #     for image in range(len(filtered_images)):
            #         x, a_3, a_2 = forward_pass(theta1, theta2, filtered_images[image])
            #         output = np.argmax(a_3) #gets largest index

            #         y = np.zeros((10,1))
            #         for i in range(len(y)):
            #             if i == filtered_labels[image]:
            #                 y[i] = 1
                    
            #         list_a_3.append(a_3)
            #         list_y.append(y)
                    
            #         if output == filtered_labels[image]:
            #             matches+=1
                
            #     accuracy = matches / len(filtered_images)
            #     print(str(epoch) + " " + str(loss(list_y, list_a_3)) + " " + str(accuracy))

        np.savetxt("f1.txt", theta1)
        np.savetxt("f2.txt", theta2)

        


