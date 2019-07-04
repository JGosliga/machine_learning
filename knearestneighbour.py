import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Generate 10 random data points
    x = np.random.rand(2,10)
    # Assign classes
    # y1 = [0,0,0,0,0,1,1,1,1,1]
    y1 = np.ndarray.tolist(np.random.rand(1,10))
    y = [round(x,0) for x in y1[0]]
    print("y = " + str(y))

    # Plot data
    plt.scatter(x[0], x[1], c=y)
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.show()

    print('Complete')