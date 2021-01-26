from main import get_significant, plot_results


def read_list_from_file(file_name):
    l = []
    with open(file_name, 'r') as f:
        for line in f:
            number = line[:-1]
            l.append(float(number))
    return l


def load_results(file_names):
    accuracies = read_list_from_file(file_names[0])
    class1_res = read_list_from_file(file_names[1])
    class2_res = read_list_from_file(file_names[2])
    return accuracies, class1_res, class2_res


if __name__ == '__main__':
    kernels = ['rbf', 'linear']
    for normalize in [True, False]:
        if normalize:
            normalized = '_normalized'
        else:
            normalized = ''
        for kernel in kernels:
            file_names = ['accuracies.txt', 'class1_posrate.txt', 'class2_posrate.txt']
            accuracies, class1_res, class2_res = load_results(file_names)
            signs = get_significant(accuracies)
            plot_results(accuracies, class1_res, class2_res, f'graph_{kernel}{normalized}.png', signs)
