import random
import matplotlib.pyplot as plt
import time

startPositionsR = [[-4500, -4400], [-4100, -3000], [-1800, -2400], [-2500, -3400], [-2000, -1400]]
startPositionsG = [[4500, -4400], [4100, -3000], [1800, -2400], [2500, -3400], [2000, -1400]]
startPositionsB = [[-4500, 4400], [-4100, 3000], [-1800, 2400], [-2500, 3400], [-2000, 1400]]
startPositionsP = [[4500, 4400], [4100, 3000], [1800, 2400], [2500, 3400], [2000, 1400]]

pointsCount = 40000
gridSize = 100

seed = 55
localRandom = random.Random(seed)

ks = [1, 3, 7, 15]
colors = ['R', 'G', 'B', 'P']
lastColor = None

class Dot:
    def __init__(self, color, position):
        self.color = color
        self.position = position

def AddPointGrid(point, grid):
    grid_x = int(point.position[0] // gridSize)
    grid_y = int(point.position[1] // gridSize)
    grid.setdefault((grid_x, grid_y), []).append(point)

def GenerateStartDots(points, grid):
    for position in startPositionsR:
        points.append(Dot('R', position))
        AddPointGrid(Dot('R', position), grid)
    for position in startPositionsG:
        points.append(Dot('G', position))
        AddPointGrid(Dot('G', position), grid)
    for position in startPositionsB:
        points.append(Dot('B', position))
        AddPointGrid(Dot('B', position), grid)
    for position in startPositionsP:
        points.append(Dot('P', position))
        AddPointGrid(Dot('P', position), grid)

    return points, grid

def GenerateRandomDotPosition(points):
    global lastColor

    availableColors = [color for color in colors if color != lastColor]
    randomColor = localRandom.choice(availableColors)
    lastColor = randomColor

    while True:
        if localRandom.random() < 0.99:
            if randomColor == 'R':
                x = localRandom.randint(-5000, 499) #x < 500, y < 500
                y = localRandom.randint(-5000, 499)
            if randomColor == 'G':
                x = localRandom.randint(-499, 5000) #x > -500, y < 500
                y = localRandom.randint(-5000, 499)
            if randomColor == 'B':
                x = localRandom.randint(-5000, 499) #x < 500, y > -500
                y = localRandom.randint(-499, 5000)
            if randomColor == 'P':
                x = localRandom.randint(-499, 5000) #x > -500, y > -500
                y = localRandom.randint(-499, 5000)
        else:
            x = localRandom.uniform(-5000, 5000)
            y = localRandom.uniform(-5000, 5000)

        newDot = Dot(randomColor, [x, y])
        if newDot.position not in [point.position for point in points]:
            return newDot

def CalculateDistance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def GetNearestNeighbors(grid, new_point, k):
    grid_x = int(new_point.position[0] // gridSize)
    grid_y = int(new_point.position[1] // gridSize)

    search_points = []
    radius = 1

    while len(search_points) < k:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1): search_points.extend(grid.get((grid_x + dx, grid_y + dy), []))
        radius += 1
        if len(search_points) >= k: break

    search_points.sort(key=lambda point: CalculateDistance(new_point.position, point.position))
    return search_points[:k]


def Classify(position, k, grid):
    newPoint = Dot('', position)
    nearest_neighbors = GetNearestNeighbors(grid, newPoint, k)

    colorVotes = {}
    for neighbor in nearest_neighbors: colorVotes[neighbor.color] = colorVotes.get(neighbor.color, 0) + 1

    newPoint.color = max(colorVotes, key=colorVotes.get)
    AddPointGrid(newPoint, grid)
    return newPoint.color


def Program(k, points, grid, pointsResults, accuracyResults):
    correctClassify = 0

    for point in points:
        votedColor = Classify(point.position, k, grid)

        if votedColor == point.color: correctClassify += 1

    accuracy = (correctClassify / (pointsCount + 20)) * 100
    accuracyResults[k] = accuracy
    pointsResults[k] = [point for cell_points in grid.values() for point in cell_points]


def main():
    pointsResults = {}
    accuracyResults = {}

    startTime = time.time()

    points, grid = GenerateStartDots([], {})
    for i in range(pointsCount): points.append(GenerateRandomDotPosition(points))

    for k in ks: Program(k, points.copy(), grid.copy(), pointsResults, accuracyResults)

    print(f"Total time: {time.time() - startTime:.2f} seconds")
    pointsResults[0] = points

    for k in ks: print(f"Accuracy for k={k}: {accuracyResults[k]:.2f}%")
    drawResult(pointsResults)

def drawResult(pointsResults):
    colors = {'R': 'red', 'G': 'green', 'B': 'blue', 'P': 'purple'}

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.flatten()

    ax = axes[0]
    xCoords = [point.position[0] for point in pointsResults[0]]
    yCoords = [point.position[1] for point in pointsResults[0]]
    colorsList = [colors[point.color] for point in pointsResults[0]]

    ax.scatter(xCoords, yCoords, color=colorsList, s=8)
    ax.set_title("Initial Field")

    ax.set_xlim(-5000, 5000)
    ax.set_ylim(-5000, 5000)

    for idx, k in enumerate(ks, start=1):
        ax = axes[idx]
        xCoords = [point.position[0] for point in pointsResults[k]]
        yCoords = [point.position[1] for point in pointsResults[k]]
        colorsList = [colors[point.color] for point in pointsResults[k]]

        ax.scatter(xCoords, yCoords, color=colorsList, s=8)
        ax.set_title(f"k={k}")

        ax.set_xlim(-5000, 5000)
        ax.set_ylim(-5000, 5000)

    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()