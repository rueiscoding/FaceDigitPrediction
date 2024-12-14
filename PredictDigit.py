import os
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class PredictDigit:
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
        count = 0
        result_array = [0] * 784
        for i in range(28):
            for j in range(28):
                if image[i][j] == "#":
                    result_array[count] = 1
                count+=1


        return result_array

    def forward_pass(theta_1, theta_2, image):
        if(len(image) != 784):
            print("ERROR")
            return
        image_numpy = np.array(image)
        bias_included = np.insert(image_numpy, 0, 1) #added bias
        x = bias_included.reshape(-1, 1)  #Reshape to (197, 1)
        z_2 = np.dot(theta_1, x)
        a_2_g = sigmoid(z_2)
        a_2 = np.insert(a_2_g, 0, 1, axis=0) #added bias of 1

        z_3 = np.dot(theta_2, a_2)
        a_3 = sigmoid(z_3)
        return a_3


    if __name__ == "__main__":
        # max n  is 451. 10% of data is 45
        n = 1000
        test_images=read_images("data/digitdata/testimages", 28, 28, n) # 28 rows with 28 columns
        test_labels=read_labels("data/digitdata/testlabels", n)
        test_images_feature = []
        
        for image in test_images:
            test_images_feature.append(image_to_feature(image))

        #load theta1 and theta2 
        theta1 = np.loadtxt("digitstheta1.txt")
        theta2 = np.loadtxt("digitstheta2.txt")

        index = 6 #pick image of choice here
        image_to_predict = test_images_feature[index]

        a_3 = forward_pass(theta1, theta2, image_to_predict)
        prediction = np.argmax(a_3)
        print_image(test_images[index])
        print("Neural Network Prediction: " + str(prediction))
        #print("Neural Network Output: " + str(a_3)) to see full a_3 array
        print("Actual Value: " + str(test_labels[index]))
