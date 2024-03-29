# Import

import numpy as np
from LinReg import LinReg
import random
import matplotlib.pyplot as plt

# Load the dataset

dataset=np.loadtxt(open("dataset.txt", "rb"), delimiter=",")
X=dataset[:,:-1]
Y=dataset[:,-1]

# LinReg

lR=LinReg()

# Useful functions

def bitstring2number(bitstring):
    """ Transform bitstring to number

    Args:
        - bitstring: A bitstring

    Returns:
        - The number given from his bits representation
    """
    l = bitstring.dot(2**np.arange(bitstring.size)[::-1])
    return l*2**(7-len(bitstring))

def unit_test_bitstring2number():
    """ Unit test for bitstring2number

    Returns:
        A boolean (True if the test is pass)
    """
    print(bitstring2number(np.array([1,0,1])) == 5*2**(7-3))

def fitness_for_sinus(bitstring):
    """ Give the fitness value of a bitstring for the sinus case

    Args:
        - bitstring: A bitstring

    Returns:
        The fitness value of the bitstring
    """
    x = bitstring2number(bitstring)
    return np.sin(x)

def unit_test_fitness_for_sinus():
    """ Unit test for fitness_for_sinus

    Returns:
        A boolean (True if the test is pass)
    """
    x = bitstring2number(np.array([1,0,1]))
    print(np.sin(x) == np.sin(5*2**(7-3)))

def fitness_for_sinus_with_penalty(bitstring,interval,penalty_rate):
    """ Give the fitness value with the penalty of a bitstring for the sinus case

    Args:
        - bitstring: A bitstring

    Returns:
        The fitness value with the penalty of the bitstring
    """
    x = bitstring2number(bitstring)
    if interval[0]<=x<=interval[1]:
        return np.sin(x)
    elif x<interval[0]:
        return np.sin(x)-penalty_rate*(interval[0]-x)
    elif x>interval[1]:
        return np.sin(x)-penalty_rate*(x-interval[1])

def population_entropy(population):
    """ Give the entropy of a given population

    Args:
        - population: A population

    Returns:
        The entropy of the population
    """
    bit_proba = np.mean(population,axis=1)
    return - np.sum(bit_proba.dot(np.log(bit_proba)))

def fitness_for_data_case(bitstring):
    """ Give the fitness value of a bitstring for the data case

    Args:
        - bitstring: A bitstring

    Returns:
        The fitness value of the bitstring
    """
    bitstring = np.array(bitstring)

    x_filtered=lR.get_columns(X,bitstring)
    return 1/lR.get_fitness(x_filtered, Y,4)

# Initiate the population

def initiate_population(n,bitstring_lentgh):
    """Initiate the population
    
    Args:
        - n: The number of individual
        - bitstring_lentgh: The lentgth of a bitstring

    Returns:
        - A random population of n individual
    """
    p=np.random.randint(2, size=(n,bitstring_lentgh))
    if p.all()==0:
        p=np.random.randint(2, size=(n,bitstring_lentgh))
    return p

# Selection

def trivial_selection(population,n,fitness_fonction):
    """Trivial selection: takes the best individual
    
    Args:
        - population: The population
        - n: The number of individual we keep
        - fitness_fonction: fitness_fonction

    Returns:
        - A population of the n best individual of the previous population and the previous population without these individual
    """
    
    F=[]
    for i,bitstring in enumerate(population):
        F.append([i,fitness_fonction(bitstring)])
    F=np.array(F)
    F=F[F[:, 1].argsort()[::-1]]
    F=F[:n,:]
    R=F[n:,:]
    F=F[:,0]
    R=R[:,0]
    F=F.astype(int)
    R=R.astype(int)
    return population[F],population[R]

# Compete

def trivial_compete(i1,i2,fitness_fonction):
    """Compete between two individual

    Args:
        - i1: An individual
        - i2: An individual
        - fitness_fonction: the fitness fonction

    Returns:
        - The best individual
    """
    i1 = np.array(i1)
    i2 = np.array(i2)

    f1=fitness_fonction(i1)
    f2=fitness_fonction(i2)

    if f1>f2:
        return i1
    else :
        return i2

# Crossover

def random_crossover(p1,p2):
    """Crossover between two parent

    Args:
        -p1: A parent
        -p2: A parent
    
    Returns:
        - Two child
    """
    c1=[]
    c2=[]
    for i in range(len(p1)):
        if random.randint(0,1)==0:
            c1.append(p1[i])
            c2.append(p2[i])
        else:
            c1.append(p2[i])
            c2.append(p1[i])
    return c1,c2

# Mutation

def normal_mutation(c,probability):
    """Mutation of an individual

    Args:
        - c: An individual
        - probability: A probability

    Returns:
        An individual with the given probability to be a mutation of c else it return c
        (only one elements of c is mutated in case of mutation)
    """
    mutate_c=c

    if random.random() < probability:
        i=random.randint(0,len(c)-1)
        if mutate_c[i]==0:
            mutate_c[i]=1
        else :
            mutate_c[i]=0
    
    return mutate_c

# Genetic algorithm class
class Ga:
    """ Genetic algorithm class 
    """
    def __init__(self,crossover_type,crossover_rate,mutation_type,mutate_rate,selection_type,fitness_fonction,crowding_option=False, 
                compete_type='none',interval='none',penalty_rate='0') -> None:
        """ Constructor of the genetic algorithm class

        Args:
            - self: The genetic algorithm
            - crossover_type: The type of the crossover
            - crossover_rate: The rate of the crossover
            - mutation_type: The type of the mutation
            - mutate_rate: The rate of the mutation
            - selection_type: The type of the selection
            - fitness_fonction: The fitness function
            - crowding_option: Option whether using crowding or not
            - compete_type: The type of competion between parent and child
            - interval: The interval of the solution space (for the sinus case)
            - penalty_rate: The penalty rate for restricting the solution to the interval (for the sinus case)

        Returns:
            The genetic algorithm with the given parameters
        """

        self.crossover_rate = float(crossover_rate)
        self.mutate_rate = float(mutate_rate)
        self.crowding_option = crowding_option

        if interval !='none':
            I = []

            L = interval.split(' ')
            I.append(int(L[0]))
            I.append(int(L[1]))

            self.interval = I
            self.penalty_rate = float(penalty_rate)
            f_sin = lambda x:fitness_for_sinus_with_penalty(x,self.interval,self.penalty_rate)
        else :
            f_sin = fitness_for_sinus

        if crossover_type == 'random':
            self.crossover = random_crossover
        
        if mutation_type == 'normal':
            self.mutation = normal_mutation

        if selection_type == 'trivial_selection':
            self.selection = trivial_selection
        
        if fitness_fonction == 'fitness_for_sinus':
            self.fitness_function = f_sin

        if fitness_fonction == 'fitness_for_data_case':
            self.fitness_function = fitness_for_data_case

        if compete_type == 'trivial_compete':
            self.compete = trivial_compete

    def one_cycle_function(self,p1,p2):
        """Crossover of parent, mutation of their child and compete between parents and childs if the crowding option is on

        Args:
            - p1: A parent
            - p2: A parent
        
        Returns:
            Individuals that result after crossover, mutation and compete
        """
        c1,c2=self.crossover(p1,p2)

        c1 = self.mutation(c1,self.mutate_rate)
        c2 = self.mutation(c2,self.mutate_rate)

        if self.crowding_option == True:

            d_p1_c1 = np.abs(np.sum(p1-c1))
            d_p2_c2 = np.abs(np.sum(p2-c2))
            d_p1_c2 = np.abs(np.sum(p1-c2))
            d_p2_c1 = np.abs(np.sum(p2-c1))

            if d_p1_c1+d_p2_c2<d_p1_c2+d_p2_c1:
                i1=self.compete(p1,c1,self.fitness_function)
                i2=self.compete(p2,c2,self.fitness_function)
            else :
                i1=self.compete(p1,c2,self.fitness_function)
                i2=self.compete(p2,c1,self.fitness_function)

            return i1,i2

        else :
            return c1,c2

    def one_cycle(self, population, number_of_individuals, number_of_parents):
        """ What happen in one cycle of the genetic algorithm

        Args:
            - population: A population
            - number_of_individuals: The number of individuals of the current population
            - number_of_parents: The number of parents selected
        
        Returns:
            The next population
        """
        childs=[]
        parents,pop_without_parent=self.selection(population,number_of_parents,self.fitness_function)
        if self.crowding_option == True:
            population = pop_without_parent
        for k in range(0,number_of_parents,2):
            c1,c2=self.one_cycle_function(parents[k],parents[k+1])
            childs.append(c1)
            childs.append(c2)
        population= np.append(population,np.array(childs),axis=0)
        if self.crowding_option == False:
            population,_=self.selection(population,number_of_individuals,self.fitness_function)

        return population

    def run(self, number_of_individuals, number_of_cycle,bitstring_length):
        """ Run the genetic algorithm

        Args:
            - population: A population
            - number_of_individuals: The number of individuals of the current population
            - number_of_parents: The number of parents selected
        
        Returns:
            All generations and the entropy
        """
        number_of_parents = int(number_of_individuals*self.crossover_rate)
        entropy = []
        populations=[]
        population=initiate_population(number_of_individuals,bitstring_length)
        populations.append(population)
        entropy.append(population_entropy(population))

        for _ in range(number_of_cycle):
            population = self.one_cycle(population, number_of_individuals,number_of_parents)
            populations.append(population)
            entropy.append(population_entropy(population))
        
        return populations,entropy

