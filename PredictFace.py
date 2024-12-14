import os
import numpy as np

# class that runs a single face from the test data set through nn and returns predicted value

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class PredictFace:
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

    def image_to_feature(image):
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

    def forward_pass(theta_1, theta_2, image):
        if(len(image) != 168):
            print("ERROR")
            return
        image_numpy = np.array(image)
        bias_included = np.insert(image_numpy, 0, 1) #added bias
        x = bias_included.reshape(-1, 1)  #Reshape to (43, 1)
        z_2 = np.dot(theta_1, x)
        a_2_g = sigmoid(z_2)
        a_2 = np.insert(a_2_g, 0, 1, axis=0) # added bias of 1

        z_3 = np.dot(theta_2, a_2)
        a_3 = sigmoid(z_3)
        return a_3


    if __name__ == "__main__":
        n = 150
        test_images=read_images("data/facedata/facedatatest", 60, 70, n)
        test_labels=read_labels("data/facedata/facedatatestlabels", n)
        test_images_feature = []
        for image in test_images:
            test_images_feature.append(image_to_feature(image))

        theta1 = np.loadtxt('facetheta1.txt') #values from face neuralnetwork
        theta2 = np.loadtxt('facetheta2.txt')

        index = 1 #pick image of choice here
        image_to_predict = test_images_feature[index]

        a_3 = forward_pass(theta1, theta2, image_to_predict)
        if a_3[0] >= 0.5:
            prediction = 1
        else:
            prediction = 0

        print_image(test_images[index])
        print("Neural Network Prediction: " + str(prediction))
        #print("Neural Network Output: " + str(a_3[0])) to see actual value of a_3[0]
        print("Actual Value: " + str(test_labels[index]))




