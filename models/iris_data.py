import time

import pandas as pd
from IPython.display import clear_output
from matplotlib import pyplot as plt
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import VQC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

if __name__ == "__main__":
    iris_data = load_iris()
    x = iris_data.data
    y = iris_data.target

    x_scaled = MinMaxScaler().fit_transform(x)

    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df["class"] = pd.Series(iris_data.target)

    # sns.pairplot(df, hue="class", palette="tab10")
    # plt.show()

    '''fix seed so result is reproducible'''
    algorithm_globals.random_seed = 123
    xtrain, xtest, ytrain, ytest = train_test_split(
        x_scaled, y, train_size=0.8, random_state=algorithm_globals.random_seed)

    svc = SVC()
    svc.fit(xtrain, ytrain)

    print(f"Classical SVC on the training dataset: {svc.score(xtrain, ytrain):.2f}")
    print(f"Classical SVC on the test dataset:     {svc.score(xtest, ytest):.2f}")

    num_features = x_scaled.shape[1]

    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    # print(feature_map.decompose())

    '''Circuit mit 16 verschiedenen Theta Variablen, die trainiert werden'''
    '''Mehr reps -> mehr gate wdhs -> mehr entanglement und mehr params -> model flexibler aber complexer'''
    ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
    # print(ansatz.decompose())

    '''Optim. Alg. gradient-free'''
    optimizer = COBYLA(maxiter=100)

    '''Wo wir trainieren -> hier Simulator'''
    sampler = Sampler()

    objective_func_vals = []
    plt.rcParams["figure.figsize"] = (12, 6)


    def callback_graph(weights, obj_func_eval):
        clear_output(wait=True)
        objective_func_vals.append(obj_func_eval)
        plt.title("Objective function value against iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(objective_func_vals)), objective_func_vals)
        plt.show()

    '''Variational Quantum Classifier'''
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        #callback=callback_graph,
    )

    # clear objective value history
    objective_func_vals = []

    start = time.time()
    vqc.fit(xtrain, ytrain)
    elapsed = time.time() - start

    print(f"Training time: {round(elapsed)} seconds")
    print(f"Quantum VQC on the training dataset: {vqc.score(xtrain, ytrain):.2f}")
    print(f"Quantum VQC on the test dataset:     {vqc.score(xtest, ytest):.2f}")








