import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

def plot_simple_graph_with_fit(data, title, xlabel, ylabel, saveToFile):
    print(f"Plotting {title} with fitted line..")

    x,y = data
     
    x_reshaped = x.reshape(-1,1)

    ransac = RANSACRegressor(LinearRegression(), random_state=42)
    ransac.fit(x_reshaped, y)
    line_y = ransac.predict(x_reshaped)
    median = np.median(y)
    print(f"Mean of the RANSAC line: {np.mean(line_y)}")
    print(f"Median of the data: {median}")

    plt.plot(x, y, '-o', color='blue', markersize=3, label='Original data')
    plt.plot(x, line_y, '-', color='red', label= 'Fitted line')
    plt.axhline(median, linestyle='--', color='green', label='Median of the data')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(saveToFile)
    plt.close()



def plot_csv_times(csv_path):

    fps = 29.97 #this is the fps of the training videos
    times = []
    mis_times = []
    with open(csv_path, newline='') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)

        for row in reader:
            _, start, min_dist, min_mis = row

            # convert to int
            min_dist = int(min_dist)
            start    = int(start)

            min_mis = int(min_mis)

            time = (min_dist / fps) - (start / fps)
            time_mis = (min_mis / fps) - (start / fps)
            times.append(time)
            mis_times.append(time_mis)

    timeline = np.arange(len(times))
    plot_simple_graph_with_fit((timeline, times),
                               title=f'Full rotation from min mean dist, median: {np.median(times):.2f}s',
                               xlabel='Video "id"',
                               ylabel='Time (s)',
                               saveToFile='full_rotation_mean_dist.png')

    plot_simple_graph_with_fit((timeline, mis_times),
                               title=f'Full rotation from min mismatch, median: {np.median(mis_times):.2f}s',
                               xlabel='Video "id"',
                               ylabel='Time (s)',
                               saveToFile='full_rotation_min_mismatch.png')

if __name__ == "__main__":
    plot_csv_times('./full_rotation_times.csv')
