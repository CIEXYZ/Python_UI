import numpy as np
import matplotlib.pyplot as plt
import mplcursors

learningRate = 0.1
epochs = 500
momentum = 0.9

class Linear:
    def __init__(self, inputSize, outputSize):
        self.weightsInput = np.random.randn(outputSize, inputSize) * np.sqrt(2 / inputSize)
        self.weightsOutput = np.zeros((outputSize, 1))
        self.gradient = None

    def Forward(self, data):
        self.data = data
        return np.dot(self.weightsInput, data) + self.weightsOutput

    def Backward(self, gradient):
        if self.gradient is None:
            self.gradient = gradient

        weightsGradient = np.dot(gradient, self.data.T)
        dataGradient = np.dot(self.weightsInput.T, gradient)

        self.gradient = momentum * self.gradient + (1 - momentum) * weightsGradient
        self.weightsInput -= learningRate * self.gradient
        self.weightsOutput -= learningRate * gradient

        return dataGradient

class Sigmoid:
    def __init__(self):
        self.output = None

    def Forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def Backward(self, gradient):
        return gradient * self.output * (1 - self.output)

class Tanh:
    def __init__(self):
        self.output = None

    def Forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def Backward(self, gradient):
        return gradient * (1 - self.output ** 2)

class ReLU:
    def Forward(self, x):
        self.out = np.maximum(0, x)
        return self.out

    def Backward(self, gradient):
        return gradient * (self.out > 0)

def Train(target, targetName):
    train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    train = np.expand_dims(train, axis=2)

    layers = [
        Linear(2, 4),
        Tanh(),
        Linear(4,4),
        ReLU(),
        Linear(4, 1),
        Sigmoid()
    ]

    losses = []
    for epoch in range(epochs):
        totalLoss = 0
        for data, prediction in zip(train, target):
            for layer in layers: data = layer.Forward(data)

            loss = (data - prediction) ** 2
            totalLoss += loss[0][0]
            gradient = data - prediction

            for layer in reversed(layers): gradient = layer.Backward(gradient)

        lossAvg = (totalLoss / len(train)) * 100
        losses.append(lossAvg)
        if (epoch + 1) % 100 == 0: print(f"{targetName} - Epoch {epoch+1}, Loss: {lossAvg:.2f}%")

    for data in train:
        result = np.array(data)
        for layer in layers: result = layer.Forward(result)
        print(f"{targetName} - Data: {data.squeeze()}, Prediction: {result.squeeze():.4f}")
    return losses

def main():
    targetXOR = np.array([[0], [1], [1], [0]])
    targetAND = np.array([[0], [0], [0], [1]])
    targetOR = np.array([[0], [1], [1], [1]])

    targets = [
        (targetXOR, "XOR"),
        (targetAND, "AND"),
        (targetOR, "OR")
    ]

    plt.figure(figsize=(12, 8))

    for target, name in targets:
        print(f"\nTraining for {name} target:")
        losses = Train(target, name)

        plt.plot(range(1, epochs + 1), losses, label=f'{name} Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss (%)')
    plt.title('Training Loss (XOR, AND, OR)')
    plt.legend()
    plt.grid(True)

    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f"Epoch: {int(sel.target[0])}\nLoss: {sel.target[1]:.2f}%"))

    plt.show()

if __name__ == '__main__':
    main()
