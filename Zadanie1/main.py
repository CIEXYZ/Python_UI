import sys
import random
import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from matplotlib import pyplot as plt
from numpy.ma.extras import average

populationSize = 200
generationsNum = 1000
elitesNum = 200

mutationChance = 0.1
crossoverChance = 0.9

rowNum = 12 #height
colNum = 10 #width
rockNum = 6
#Rocks positions:
rocks = [[1, 2], [2, 4], [4, 3], [8, 6], [9, 6], [5, 1]]
positions = []

genNum = rowNum + colNum + rockNum
population = []

field = np.zeros((colNum, rowNum))
clearedField = np.zeros((colNum, rowNum))
clearedNum = 1 #pre UI

bestPerson = None
bestClearedField = None

#Turns
# -1 - Left
# 1 - Right

x = 0
y = 0

#Generovanie vsetkych pozicii na okraji pola
def generateRandomPosition():
    for i in range(colNum):
        positions.append((0, i))
        positions.append((rowNum - 1, i))

    for i in range(rowNum):
        if (i, 0) not in positions: positions.append((i, 0))
        positions.append((i, colNum - 1))

#Zoberieme random poziciu z array positions
def getRandomPosition(rpositions):
    randPosition = []
    for _ in range(28):
        selected_position = random.choice(rpositions)
        randPosition.append(selected_position)
        rpositions.remove(selected_position)
    return randPosition

#Generujeme populaciu a nahodne geny
def generatePopulation():
    population.clear()
    generateRandomPosition()

    for _ in range(populationSize):
        rpositions = getRandomPosition(positions.copy())
        turns = [random.choice([-1, 1]) for _ in range(28)]
        population.append([rpositions, turns])

#Generujeme ciste pole a kamene
def generateField():
    field.fill(0)
    for rock in rocks:
        field[rock[1], rock[0]] = -1

#Ideme, a kontrolujeme ci mozeme ist dopredu
def moving(dx, dy):
    cleared = 0
    global x, y
    global clearedField

    while True:
        if 0 <= x < rowNum and 0 <= y < colNum:
            if clearedField[y][x] == -1 or clearedField[y][x] >= 1: break
            elif clearedField[y][x] == 0:
                clearedField[y][x] = clearedNum
                cleared += 1
        else: break

        x += dx
        y += dy

        if not(0 <= x < rowNum and 0 <= y < colNum): break
        elif clearedField[y][x] == -1 or clearedField[y][x] >= 1: break
    return cleared

#Ak sme na okraji, zistime ci ist hore/dole, alebo napravo/nalavo
def changeDirection(x, y):
    if y == 0: return 0, 1
    if x == 0: return 1, 0
    if y == colNum - 1: return 0, -1
    if x == rowNum - 1: return -1, 0

#Simulacia hrabania, menime strany kam ist
def simulateActions(person):
    global x, y
    global field
    global clearedField
    global clearedNum

    positions, turns = person
    clearedField = field.copy()
    cleared = 0
    clearedNum = 1

    x, y = positions[0]
    dx, dy = changeDirection(x, y)

    i, j = 1, 0
    while j < len(turns):
        if not (0 <= x < rowNum and 0 <= y < colNum):
            if i >= 28: break

            x, y = positions[i]
            if clearedField[y][x] == -1 or clearedField[y][x] >= 1: return 0

            i += 1
            clearedNum += 1
            dx, dy = changeDirection(x, y)
        if clearedField[y][x] == -1 or clearedField[y][x] >= 1:
            x -= dx
            y -= dy

            turn = turns[j]
            j += 1

            if turn == -1: newdx, newdy = dy, -dx
            else: newdx, newdy = -dy, dx

            if (0 <= x + newdx < rowNum and 0 <= y + newdy < colNum):
                if clearedField[y + newdy][x + newdx] != -1 or clearedField[y + newdy][x + newdx] == 0: dx, dy = newdx, newdy
                else:
                    if turn == -1: newdx, newdy = dy, -dx
                    else: newdx, newdy = -dy, dx

                    if (0 <= x + newdx < rowNum and 0 <= y + newdy < colNum) and (clearedField[y + newdy][x + newdx] != -1 or clearedField[y + newdy][x + newdx] == 0): dx, dy = newdx, newdy
                    else: return 0

            x += dx
            y += dy

        cleared += moving(dx, dy)
    return cleared

def fitness(person):
    global clearedNum
    goal = 114
    result = simulateActions(person)
    clearedNum = 0
    #print(f'Cleared cells: {result}')

    return result / goal

def selectionRoulette(population, fitnessValues):
    totalFitness = sum(fitnessValues)
    probabilities = [fitnessValues / totalFitness for fitnessValues in fitnessValues]
    selectedPopulation = []

    for i in range(len(population)):
        selectedPopulation.append(random.choices(population, weights=probabilities)[0].copy())
    return selectedPopulation.copy()

def selectionTournament(population, fitnessValues):
    selectedPopulation = []
    for i in range(len(population)):
        contestants = random.sample(list(enumerate(population.copy())), k=3)
        contestants.sort(key=lambda x: fitnessValues[x[0]], reverse=True)

        selectedPopulation.append(contestants[0][1])
    return selectedPopulation.copy()

def crossover(parent1, parent2):
    crossoverPoint = random.randint(1, len(parent1[0]) - 1)
    child1Positions = parent1[0][:crossoverPoint] + parent2[0][crossoverPoint:]
    child2Positions = parent2[0][:crossoverPoint] + parent1[0][crossoverPoint:]

    crossoverPoint = random.randint(1, len(parent1[1]) - 1)
    child1Turns = parent1[1][:crossoverPoint] + parent2[1][crossoverPoint:]
    child2Turns = parent2[1][:crossoverPoint] + parent1[1][crossoverPoint:]

    return [child1Positions, child1Turns], [child2Positions, child2Turns]

def mutate(person):
    for i in range(len(person[0])):
        if random.random() < mutationChance:
            person[0][i] = random.choice(positions)

    for i in range(len(person[1])):
        if random.random() < mutationChance: person[1][i] = random.choice([-1, 1])

    return person.copy()

#Vizualizacia pohrabanej zahrady
def initUI(self):
    grid = QGridLayout()
    grid.setSpacing(0)
    self.setLayout(grid)

    for i in range(10):
        for j in range(12):
            button = QPushButton(str(field[i][j]))
            button.setFixedSize(60, 60)
            button.setDisabled(True)
            if bestClearedField[i][j] == -1:
                button.setObjectName('stone')
            elif bestClearedField[i][j] > 0:
                button.setObjectName('cleared')
                button.setText(str(int(bestClearedField[i][j])))
            else: button.setObjectName('emptyField')
            grid.addWidget(button, i, j)
    self.setWindowTitle('UI_1a')
    initStyleSheet(self)
    self.show()

#Colors a ine veci pre vizualizaciu
def initStyleSheet(self):
    file = QFile('defaultField.qss')
    file.open(QFile.ReadOnly | QFile.Text)
    stylesheet = file.readAll().data().decode('utf-8')
    self.setStyleSheet(stylesheet)

#Vizualizacia pohrabanej zahrady v console
def printField():
    for row in bestClearedField:
        print(' '.join(f'{int(val):2}' for val in row))

#Geneticky algorithmus, dostavame najlepseho jedinca a nakonci ukazujeme graf
def genetic_algorithm():
    global population
    global bestPerson
    global bestClearedField

    app = QApplication(sys.argv)

    bestFitnessGenerations = []
    avgFitnessGenerations = []

    generatePopulation()

    for generation in range(generationsNum):
        generateField()

        print(f"Generation {generation + 1}")

        fitnessValues = [fitness(person) for person in population]

        bestFitnessGenerations.append(max(fitnessValues))
        avgFitnessGenerations.append(average(fitnessValues))

        print(f"Average fitness: {average(fitnessValues) * 100:.2f}%")
        print(f"Best fitness: {max(fitnessValues) * 100:.2f}%")

        if not bestPerson or max(fitnessValues) > fitness(bestPerson):
            bestPerson = population[fitnessValues.index(max(fitnessValues)) ]
            bestClearedField = clearedField.copy()

        if max(fitnessValues) * 100 >= 100.00: break

        elites = [person for person, _ in sorted(zip(population, fitnessValues), key=lambda x: x[1], reverse = True)][:elitesNum]

        selectedPopulation = selectionRoulette(population.copy(), fitnessValues) #Mame moznost ti vybrat selection, len vymenime funkcie
        offset = []

        for parent1, parent2 in zip(selectedPopulation[::2], selectedPopulation[1::2]):
            if random.random() < crossoverChance: parent1, parent2 = crossover(parent1.copy(), parent2.copy())
            offset.append(parent1.copy())
            offset.append(parent2.copy())

        for i in range(populationSize):
            if random.random() < mutationChance: offset[i] = mutate(offset[i].copy())

        for i in range(elitesNum):
            offset[random.sample(range(populationSize), elitesNum)[i]] = elites[i]

        population = offset.copy()

    printField()
    window = QWidget()
    initUI(window)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(bestFitnessGenerations) + 1), bestFitnessGenerations, label='Best Fitness')
    plt.plot(range(1, len(bestFitnessGenerations) + 1), avgFitnessGenerations, label='Average Fitness')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()

    return bestPerson

def main():
    genetic_algorithm()

if __name__ == '__main__':
    main()