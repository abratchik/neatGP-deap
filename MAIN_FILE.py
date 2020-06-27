import argparse
import copy
import csv
import operator
import os
import sys
import time

import numpy as np
import yaml
from deap import base
from deap import creator
from deap import gp
from deap import tools

import eaneatGP
import gp_conf as neat_gp
import init_conf
from conf_primitives import conf_sets, vector_benchmarks

config = []
toolbox = None


def eval_symb_reg(individual, points):
    points.flags['WRITEABLE'] = False
    func = toolbox.compile(expr=individual)
    if config["benchmark"]:
        vector = vector_benchmarks(config["problem"], points)
        data_x = copy.deepcopy(np.asarray(points)[:])
    else:
        vector = copy.deepcopy(points[config["num_var"]])
        data_x = copy.deepcopy(np.asarray(points)[:config["num_var"]])

    vector_x = func(*data_x)
    with np.errstate(divide='ignore', invalid='ignore'):
        if isinstance(vector_x, np.ndarray):
            for e in range(len(vector_x)):
                if np.isnan(vector_x[e]) or np.isinf(vector_x[e]):
                    vector_x[e] = 0.
    result = np.sum((vector_x - vector) ** 2)
    return np.sqrt(result / len(points[0])),


def train_test(n_corr, p, problem):
    database_name = config["database_name"]
    n_archivot = './data_corridas/%s/test_%d_%d.txt' % (problem, p, n_corr)
    n_archivo = './data_corridas/%s/train_%d_%d.txt' % (problem, p, n_corr)
    if not (os.path.exists(n_archivo) or os.path.exists(n_archivot)):
        direccion = "./data_corridas/%s/%s" % (problem, database_name)
        with open(direccion) as spambase:
            spamReader = csv.reader(spambase, delimiter=' ', skipinitialspace=True)
            num_c = sum(1 for line in open(direccion))
            num_r = len(next(csv.reader(open(direccion), delimiter=' ', skipinitialspace=True)))
            Matrix = np.empty((num_r, num_c,))
            for row, c in zip(spamReader, list(range(num_c))):
                for r in range(num_r):
                    try:
                        Matrix[r, c] = row[r]
                    except ValueError:
                        print('Line {r} is corrupt', r)
                        break
        if not os.path.exists(n_archivo):
            long_train = int(len(Matrix.T) * .7)
            data_train1 = Matrix.T[np.random.choice(Matrix.T.shape[0], long_train, replace=False)]
            np.savetxt(n_archivo, data_train1, delimiter=",", fmt="%s")
        if not os.path.exists(n_archivot):
            long_test = int(len(Matrix.T) * .3)
            print("long_test %s" % (long_test,))
            data_test1 = Matrix.T[np.random.choice(Matrix.T.shape[0], long_test, replace=False)]
            np.savetxt(n_archivot, data_test1, delimiter=",", fmt="%s")
    with open(n_archivo) as spambase:
        spamReader = csv.reader(spambase, delimiter=',', skipinitialspace=True)
        num_c = sum(1 for line in open(n_archivo))
        num_r = len(next(csv.reader(open(n_archivo), delimiter=',', skipinitialspace=True)))
        Matrix = np.empty((num_r, num_c,))
        for row, c in zip(spamReader, list(range(num_c))):
            for r in range(num_r):
                try:
                    Matrix[r, c] = row[r]
                except ValueError:
                    print('Line {r} is corrupt train', r)
                    break
        data_train = Matrix[:]
    with open(n_archivot) as spambase:
        spamReader = csv.reader(spambase, delimiter=',', skipinitialspace=True)
        num_c = sum(1 for line in open(n_archivot))
        num_r = len(next(csv.reader(open(n_archivot), delimiter=',', skipinitialspace=True)))
        Matrix = np.empty((num_r, num_c,))
        for row, c in zip(spamReader, list(range(num_c))):
            for r in range(num_r):
                try:
                    Matrix[r, c] = row[r]
                except ValueError:
                    print('Line {r} is corrupt test', r)
                    break
        data_test = Matrix[:]
    return data_train, data_test


def init(problem):
    global toolbox, config

    p_config = yaml.safe_load(open("./data_corridas/%s/conf.yaml" % (problem,)))
    config.update(p_config)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("FitnessTest", base.Fitness, weights=(-1.0,))
    creator.create("Individual", neat_gp.PrimitiveTree, fitness=creator.FitnessMin, fitness_test=creator.FitnessTest)

    toolbox = base.Toolbox()

    neat_cx = config["neat_cx"]
    tournament_size = config["tournament_size"]
    num_var = config["num_var"]
    pset = conf_sets(num_var)

    if neat_cx:
        toolbox.register("expr", gp.genFull, pset=pset, min_=0, max_=3)
    else:
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=7)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", init_conf.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", neat_gp.cxSubtree)
    if neat_cx:
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    else:
        toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=7)
    toolbox.register("mutate", neat_gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main(n_corr, p, problem, ngen):
    pop_size = config["population_size"]
    cxpb = config["cxpb"]  # 0.9
    mutpb = config["mutpb"]  # 0.1
    neat_cx = config["neat_cx"]

    params = config["params"]

    neat_alg = config["neat_alg"]
    neat_pelit = config["neat_pelit"]
    neat_h = config["neat_h"]
    beta = config["neat_beta"]

    data_train, data_test = train_test(n_corr, p, problem)

    toolbox.register("evaluate", eval_symb_reg, points=data_train)
    toolbox.register("evaluate_test", eval_symb_reg, points=data_test)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(3)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    eaneatGP.neat_GP(pop, toolbox, cxpb, mutpb, ngen, neat_alg, neat_cx, neat_h, neat_pelit, n_corr, p,
                     params, problem, beta, stats=mstats, halloffame=hof, verbose=True)


if __name__ == "__main__":

    config = yaml.safe_load(open("conf/conf.yaml"))

    arg_parser = argparse.ArgumentParser(description="Implementation of neat-GP on DEAP framework. ")
    arg_parser.add_argument("-g", "--generations", type=int, default=config["generations"],
                            help="Number of generations")
    arg_parser.add_argument("-b", "--run_begin", type=int, default=config["run_begin"], help="Start from iteration")
    arg_parser.add_argument("-e", "--run_end", type=int, default=config["run_end"], help="End with iteration")
    arg_parser.add_argument("problem", nargs=1, choices=["Housing", "EnergyCooling", "BreastCancer", "SMT"],
                            help="Problem")

    try:
        args = arg_parser.parse_args()
    except:
        arg_parser.print_help()
        sys.exit(0)

    ngen = args.generations
    problem = args.problem[0]

    p = config["n_problem"]

    n_corr = args.run_begin

    init(problem)

    while n_corr <= args.run_end:
        begin_p = time.time()
        main(n_corr, p, problem, ngen)
        n_corr += 1
        end_p = time.time()
