import scipy.io, os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from scipy import stats


def load_data(file_prefix, class_1_indices, class_2_indices, normalize):
    time_points = [[] for x in range(1301)]
    labels = [[] for x in range(1301)]
    for index in class_1_indices + class_2_indices:
        for file in os.listdir(f'{file_prefix}{index}'):
            data = scipy.io.loadmat(f'{file_prefix}{index}/{file}')
            arr = data['F']

            if normalize:
                arr = 2. * (arr.transpose() - np.min(arr, axis=1)) / np.ptp(arr, axis=1) - 1
                arr = arr.transpose()

            for column in range(arr.shape[1]):
                time_points[column].append(arr[:, column])
                labels[column].append(1 if index in class_1_indices else 2)
    time_points = [np.asarray(time_point) for time_point in time_points]
    labels = [np.asarray(label) for label in labels]
    print(time_points[0].shape)
    return time_points, labels


def get_significant(accuracies):
    significant = []
    for accuracy in accuracies:
        p = stats.binom_test(int(480*accuracy), n=480, p=0.5, alternative='greater')
        if p < 0.05:
            significant.append(True)
        else:
            significant.append(False)
    return significant


def running_average(value_list, interval):
    averages = [sum(value_list[i-interval:i])/interval for i in range(interval, len(value_list))]
    initial_averages = [sum(value_list[:i])/i for i in range(1, interval)]
    averages = initial_averages + averages
    return averages


def train_loocv(x_set, y_set, kernel='rbf'):
    accuracies = []
    class1_results = []
    class2_results = []
    for i, time_step in enumerate(x_set):
        if (i % 5) == 0:
            results = []
            c1_res = []
            c2_res = []
            loo = LeaveOneOut()
            for train_index, test_index in loo.split(time_step):
                X_train, X_test = time_step[train_index], time_step[test_index]
                y_train, y_test = y_set[i][train_index], y_set[i][test_index]
                model = SVC(C=1, kernel=kernel, degree=3)
                model.fit(X_train, y_train)
                # evaluate model
                yhat = model.predict(X_test)

                # store
                results.append(1 if (yhat == y_test) else 0)
                if y_test == 1:
                    c1_res.append(1 if yhat == y_test else 0)
                elif y_test == 2:
                    c2_res.append(1 if yhat == y_test else 0)

            print(f'timestep {i}')
            accuracy = sum(results) / len(results)
            class1_acc = sum(c1_res)/len(c1_res)
            class2_acc = sum(c2_res)/len(c2_res)

            print(accuracy)
            accuracies.append(accuracy)
            class1_results.append(class1_acc)
            class2_results.append(class2_acc)

    print(accuracies)
    return accuracies, class1_results, class2_results


def plot_results(accuracies, class1_results, class2_results, file_name, signs=None):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_title('SVM classifiers accuracy over time', fontsize=20)
    ax.set_ylabel('Accuracy', fontsize='13')
    ax.set_xlabel('Time from image onset [ms]', fontsize='13')
    color = 'salmon'
    ax.plot(np.linspace(-100, 1201, len(accuracies)), accuracies, label='Accuracy', color=color)
    averages = running_average(accuracies, 10)
    if signs is not None:
        for x in range(len(accuracies)):
            if signs[x]:
                ax.plot(np.asarray(x*5-100), accuracies[x], markersize=5, color=color, marker='o')
                if x != (len(signs)-1):
                    if signs[x] and signs[x+1]:
                        ax.plot([x * 5 - 100, (x+1)*5 - 100], [accuracies[x], accuracies[x+1]], linewidth=5, color=color, marker='o')
    ax.plot(np.linspace(-100, 1201, len(averages)), averages, label='Running average', color='firebrick')
    ax.plot([0, 0], [min(accuracies), max(accuracies)], color='silver', label='Image onset')
    ax.plot([500, 500], [min(accuracies), max(accuracies)], color='black', label='Image offset')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=8)

    # ax.plot(np.linspace(-100, 1201, len(accuracies)), class1_results, label='Class 1 positive rate')
    # ax.plot(np.linspace(-100, 1201, len(accuracies)), class2_results, label='Class 2 positive rate')

    ax.legend(fontsize=12)
    plt.savefig(file_name)
    plt.show()


def write_list(list, file_name):
    with open(file_name, 'w') as f:
        for acc in list:
            f.write('%s\n' % acc)


if __name__ == '__main__':
    class_one_indices = [x for x in range(13, 25)]
    class_two_indices = [x for x in range(73, 77)] + [x for x in range(78, 81)] + [x for x in range(83, 86)] + \
                        [88, 89]
    assert len(class_two_indices) == 12
    assert len(class_one_indices) == 12

    kernels = ['rbf', 'linear']
    for normalize in [True, False]:
        if normalize:
            normalized = '_normalized'
        else:
            normalized = ''
        x_set, y_set = load_data('./subj01/sess01/cond00', class_one_indices, class_two_indices, normalize)
        for kernel in kernels:
            accuracies, class_one_res, class_two_res = train_loocv(x_set, y_set, kernel)
            write_list(accuracies, f'accuracies_{kernel}{normalized}.txt')
            write_list(class_one_res, f'class1_posrate_{kernel}{normalized}.txt')
            write_list(class_two_res, f'class2_posrate_{kernel}{normalized}.txt')


