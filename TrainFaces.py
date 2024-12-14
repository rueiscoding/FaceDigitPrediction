import os
import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class TrainFaces:
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
        subgrids = []
        rows = 70
        cols = 60
        subgrid_rows = 5
        subgrid_cols = 5

        result_array = [0] * 168
        count = 0
        for i in range(0, rows, subgrid_rows):
            for j in range(0, cols, subgrid_cols):

                for i_increase in range(5):
                    for j_increase in range(5):
                        if image[i+i_increase][j+j_increase] == "#":
                            result_array[count] = 1

                count+=1

        return result_array

    def initalize_weights():
        theta_1 = np.random.uniform(-1, 1, size=(168, 169))
        theta_2 = np.random.uniform(-1, 1, 169)

        for i in range(168):
            for j in range(169):
                while(theta_1[i][j] == 0):
                    theta_1[i][j] = random.uniform(-1, 1)
        
        for k in range(169):
            while (theta_2[k] == 0):
                theta_2[k] = random.uniform(-1, 1)

        return theta_1, theta_2
    
    def forward_pass(theta_1, theta_2, image):
        if(len(image) != 168):
            print("ERROR")
            return
        image_numpy = np.array(image)
        bias_included = np.insert(image_numpy, 0, 1) #added bias
        x = bias_included.reshape(-1, 1)  #Reshape to (43, 1)
        z_2 = np.dot(theta_1, x)
        a_2_g = sigmoid(z_2)
        a_2 = np.insert(a_2_g, 0, 1, axis=0) #added bias of 1

        z_3 = np.dot(theta_2, a_2)
        a_3 = sigmoid(z_3)
        return x, a_3, a_2
    
    def compute_errors(a_3, a_2, y, theta1, theta2):
        transpose_theta1 = np.transpose(theta1)
        #subset = theta2[:, 1:]
        transpose_theta2 = np.transpose(theta2)
        delta_3 = a_3 - y
        #first_half = np.dot(transpose_theta1, delta_3)

        #including bias
        first_half = transpose_theta2[1:] * delta_3
        one_minus_a2 = 1 - a_2[1:] 
        second_half = a_2[1:] * one_minus_a2 
        reshape_first_half = first_half.reshape((168, 1))
        delta_2 = reshape_first_half * second_half

        # print("a_3 : " + str(a_3.shape))
        # print("a_2 : " + str(a_2.shape))
        # print("theta1 : " + str(theta1.shape))
        # print("theta2 : " + str(theta2.shape))
        # print("delta_2 : " + str(delta_2.shape))
        # print("delta_3 : " + str(delta_3.shape))
        # print("transpose theta2 : " + str(transpose_theta2.shape))
        # print("first half : " + str(first_half.shape))
        # print("second half : " + str(second_half.shape))

        return delta_2, delta_3

    def compute_gradient(x, a_3, a_2, delta_2, delta_3, theta1_delta, theta2_delta):

        local_theta2_delta = np.transpose(a_2 * delta_3)
        # theta1_delta = np.dot(x[1:], np.transpose(delta_2))
        #theta1_delta = np.dot(delta_2, np.transpose(x)) #gives right shape but is wrong formula
        local_theta1_delta = np.dot(delta_2, np.transpose(x))

        theta1_delta += local_theta1_delta

        theta2_delta = np.reshape(theta2_delta, (1, 169))
        theta2_delta += local_theta2_delta

        return theta1_delta, theta2_delta

    def regularize_gradient(theta1, theta2, theta1_delta, theta2_delta, iterations):
        lambda_reg = 0.2
        reg_theta1 = theta1_delta / iterations
        reg_theta2 = theta2_delta / iterations

        for val in range(reg_theta2.shape[0]):
            if val != 0: #ignore bias weight
                reg_theta2[val] += lambda_reg * theta2[val]
        
        for i in range(reg_theta1.shape[0]):
            for j in range(reg_theta1.shape[1]):
                if j != 0: #ignore bias weight
                    reg_theta1[i][j] += lambda_reg * theta1[i][j]
        
        return reg_theta1, reg_theta2


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

        n = 450
        training_images=read_images("data/facedata/facedatatrain", 60, 70, n) # 70 rows with 60 columns
        training_labels=read_labels("data/facedata/facedatatrainlabels", n)
        training_images_feature = []
        for image in training_images:
            training_images_feature.append(image_to_feature(image))

        percent = 1
        selected_images, selected_labels = get_random_sample(percent, training_images_feature, training_labels)

        theta1, theta2 = initalize_weights()
        for epoch in range(500):
            theta1_delta = np.zeros((168, 169))
            theta2_delta = np.zeros(169)

            for image in range(len(selected_images)):
                #print(image)
                x, a_3, a_2 = forward_pass(theta1, theta2, selected_images[image])
                delta_2, delta_3 = compute_errors(a_3, a_2, selected_labels[image], theta1, theta2)
                theta1_delta, theta2_delta = compute_gradient(x, a_3, a_2, delta_2, delta_3, theta1_delta, theta2_delta)
        
            reg_theta1, reg_theta2 = regularize_gradient(theta1, theta2, theta1_delta, theta2_delta, len(selected_images))
            #update the weight gradients
            theta1 = theta1 - (0.02*reg_theta1) #0.015 was last best for full data
            theta2 = theta2 - (0.02*reg_theta2)


            matches = 0

            print(epoch)
            # for image in range(len(training_images_feature)):
            #     x, a_3, a_2 = forward_pass(theta1, theta2, training_images_feature[image])
            #     if a_3[0] >= 0.5:
            #         output = 1
            #     else:
            #         output = 0
                
            #     if output == training_labels[image]:
            #         matches+=1
            
            # accuracy = matches / len(training_images_feature)
            # print(str(epoch) + " " + str(average_abs) + " " + str(accuracy))

        np.savetxt("1_1.txt", theta1)
        np.savetxt("1_2.txt", theta2)
        


