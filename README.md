<h1>Implementation of neat-GP on DEAP framework</h1>

Hi everyone! 
This is the code to implement neat-GP on python-deap, to install it you have to clone 
the repository. The example to run it is the EnergyCooling file.
The previous software that you'll need are: 
<ul>
<li>Python 3.6  https://www.python.org/downloads/</li>
<li>Deap 1.3.1 https://github.com/deap/deap</li>
<li>numpy http://www.numpy.org/ </li>
</ul>

[Apr/2016]<b>New Status:</b><br>
There's a modification on crossover and mutation, previously we could make a 
crossover AND mutation to the same individual, however we modified the algorithm 
to do it like a standard GP, where the individual pass to the crossover OR mutation 
given a probability. <br><br>
[Jun/2017]<b>New Update [Thanks to Aditya Rawal]:</b><br>
There's a modification on measure_tree.py file on the compare tree method. The method 
was not calculating the correct 'structure share' between two trees.

By the way, we made a new version of the algorithm where we integrate a local search 
method into neat-GP, you can found it in https://github.com/saarahy/NGP-LS 
(Article: http://dl.acm.org/citation.cfm?id=2931659).

<h2>Instructions</h2>
After the installation you only have to configure the parameters in the config files 
(conf.yaml) - conf/conf.yaml is for global parameters and 
data_corridas/&lt;MODEL&gt;/conf.yaml and the run the MAIN_FILE.py with the parameter 
specifying the MODEL.<br>

If you want to add or remove the primitives set you have to modify the conf_primitives.py 
file, also in this file you can check if the number of arguments that you are going 
to need is in dictionary of the rename_arguments method.

<h2>NEAT implementation</h2>
This algorithm reproduce the simplest evolutionary algorithm as presented in 
chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param neat_alg: wheter or not to use species stuff.
    :param neat_cx: wheter or not to use neatGP cx
    :param neat_h: indicate the distance allowed between each specie
    :param neat_pelit: probability of being elitist, it's used in the neat cx and mutation
    :param n_corr: run number just to wirte the txt file
    :param num_p: problem number just to wirte the txt file
    :param params:indicate the params for the fitness sharing, the diffetent
                    options are:
                    -DontPenalize(str): 'best_specie' or 'best_of_each_specie'
                    -Penalization_method(int):
                        1.without penalization
                        2.penalization fitness sharing
                        3.new penalization
                    -ShareFitness(str): 'yes' or 'no'
    :param problem: (str) name of the problem.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population.

    It uses :math:`\lambda = \kappa = \mu` and goes as follow.
    It first initializes the population (:math:`P(0)`) by evaluating
    every individual presenting an invalid fitness. Then, it enters the
    evolution loop that begins by the selection of the :math:`P(g+1)`
    population. Then the crossover operator is applied on a proportion of
    :math:`P(g+1)` according to the *cxpb* probability, the resulting and the
    untouched individuals are placed in :math:`P'(g+1)`. Thereafter, a
    proportion of :math:`P'(g+1)`, determined by *mutpb*, is
    mutated and placed in :math:`P''(g+1)`, the untouched individuals are
    transferred :math:`P''(g+1)`. Finally, those new individuals are evaluated
    and the evolution loop continues until *ngen* generations are completed.
    Briefly, the operators are applied in the following order ::

        evaluate(population)
        for i in range(ngen):
            offspring = select(population)
            offspring = mate(offspring)
            offspring = mutate(offspring)
            evaluate(offspring)
            population = offspring

    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.

<h2>Credits</h2>
Author of the original implementation: juarez.s.perla@gmail.com <br>
Refactored by: abratchik@snap2cart.com
