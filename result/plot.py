import matplotlib.pyplot as plt

def get_runtime():
    with open('runtime1.txt','r') as f:
        lines = f.readlines()
        runtimes = [[], []]  # runtimes[0] for alexnet, runtimes[1] for vgg16
        current_runtime = runtimes[0]
        for line in lines:
            line = line.strip()
            if line:  # If line is not empty
                current_runtime.append(float(line))
            else:  # If line is empty
                current_runtime = runtimes[1] if current_runtime is runtimes[0] else runtimes[0]
        return runtimes

def plot_runtime():
    runtimes = get_runtime()
    for i, runtime in enumerate(runtimes):
        model = 'alexnet' if i == 0 else 'vgg16'
        plt.figure()
        runtime = runtime[1:]  # Remove the first data point
        plt.bar(range(1, len(runtime) + 1), runtime)
        min_index = runtime.index(min(runtime)) + 1  # Find the index of the minimum data point
        offset = max(runtime) * 0.05  # Set the offset to 5% of the maximum runtime
        plt.scatter(min_index, min(runtime) + offset, color='yellow', edgecolors='black', marker='*', s=100)  # Add a yellow star marker at the minimum data point with a black edge
        plt.xlabel('cut layer')
        plt.ylabel('runtime(s)')
        plt.title(f' inference response of {model}')
        plt.savefig(f'{model}_result.png')
    plt.show()

if __name__ == '__main__':
    plot_runtime()