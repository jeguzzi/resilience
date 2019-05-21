# On the Impact of Uncertainty for Path Planning
Supplementary material, code and results for the ICRA 2019 paper by

*Jérôme Guzzi, R. Omar Chavez-Garcia, Luca M. Gambardella, Alessandro Giusti*, 
Dalle Molle Institute for Artificial Intelligence (IDSIA), USI-SUPSI, Lugano



### Abstract

We consider the problem of planning paths on graphs with some edges whose traversability is uncertain; for each uncertain edge, \mdiff{we are given a probability of being traversable (e.g., by a learned classifier)}.  We categorize different interpretations of the problem that are meaningful for mobile robots navigating partially-known environments, each of which yields a different formalization; we then focus on the case in which the true traversability of an edge is revealed only when the agent visits one of its endpoints (_Canadian Traveller Problem_).  In this context, we design a large simulation campaign on synthetic and real-world maps to study the impact of two different factors: the planning strategy, and the amount of uncertainty (which could depend on the quality of the classifier producing traversability estimates).  

## Experimental results

You find all experimental results in folder `results`.

## How to reproduce the experiments reported in the paper

The simplest way to reproduce the experiments is to use the provided docker image through docker-compose.

```bash
docker-compose pull
docker-compose up experiments
```

Running the whole experiments could take weeks depending on the number of cores.

To reduce the number of samples, use  experiments/ral_mini.j2 or modify experiments/ral.j2, in particular set

- `random_graph_number`: The number of random graph to sample (for each experiment)
- `real_map_classifier_samples`: How many samples to draw for each classifier and each real map realization.

in

```
{% set random_graph_number = 100000 -%}
{% set real_map_classifier_samples = 100 -%}
```

To modify the number of assigned cores, set the `pool` parameter in docker-compose.yaml
```yaml
    command:  python3 code/main.py --config experiments/ral.j2 --pool <NUMBER_OF_CORES>
```

## Notebook

We provide a jupyter notebook to illustrate the experiments.


```bash
docker-compose up notebook
```

and open a [browser window](http://localhost:8889/notebooks/Experiments.ipynb).


## ICRA Poster

Take a look at the poster presented at the interactive session of ICRA 2019.

![https://github.com/jeguzzi/resilience/blob/master/ICRA_2019_Poster.pdf](https://github.com/jeguzzi/resilience/blob/master/ICRA_2019_Poster.pdf "Get the pdf.")



