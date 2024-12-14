import os
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class TestFaces:
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
    # def image_to_feature(image):
    #     subgrids = []
    #     rows = 70
    #     cols = 60
    #     subgrid_rows = 5
    #     subgrid_cols = 5

    #     for i in range(0, rows, subgrid_rows):
    #         for j in range(0, cols, subgrid_cols):
    #             subgrid = []
    #             for k in range(i, min(i + subgrid_rows, rows)):
    #                 subgrid.append(image[k][j:min(j + subgrid_cols, cols)])
    #             subgrids.append(subgrid)

    #     result_array = [0] * 168

    #     # Iterate over each small grid and check if any pixel is marked as "#"
    #     for index, grid in enumerate(subgrid):
    #         if any('#' in row for row in grid):
    #             result_array[index] = 1

    #     return result_array

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
                            #print(count)

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
        # max n  is 451. 10% of data is 45
        n = 150
        test_images=read_images("data/facedata/facedatatest", 60, 70, n) # 70 rows with 60 columns
        test_labels=read_labels("data/facedata/facedatatestlabels", n)
        test_images_feature = []
        for image in test_images:
            test_images_feature.append(image_to_feature(image))

        theta1 = np.loadtxt('1_1.txt')
        theta2 = np.loadtxt('1_2.txt')

        matches = 0

        for image in range(len(test_images_feature)):
            a_3 = forward_pass(theta1, theta2, test_images_feature[image])
            print(str(a_3[0]) + " " + str(test_labels[image]))

            if a_3[0] >= 0.5:
                output = 1
            else:
                output = 0
            
            if output == test_labels[image]:
                matches+=1
        
        print(matches)
        accuracy = matches / len(test_images_feature)
        print(accuracy)




