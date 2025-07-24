+++
title = 'Metrobike Optimization Around UT Austin'
date = 2024-12-10T18:47:40-06:00
author = ["Hasith Vattikuti", "Dev Desai", "Eric Liang", "Viren Govin"]
draft = false
categories = ["projects"]
tags = ["optimization", "agent based modeling", "python"]
showtoc = true
+++
> *Original post can be found: [here](https://hasithv.github.io/posts/24-12-10-metrobike/)*

This project was done as our final project for [William Gilpin's](https://www.wgilpin.com/) Graduate [Computational Physics Course](https://www.wgilpin.com/cphy/?utm_source=en_us_srepgw). Our complete GitHub repository, with instructions on how to replicate our results, can be found [here](https://github.com/devddesai/metrobike).

# Introduction
The goal of this project is to simulate the behavior of a bike-sharing system in a network of stations and destinations, and then optimize the positions of the stations. We approach the simulation of the bike-sharing system with Agent Based Modeling (ABM).

Ultimately, we aim to find the optimal locations for [metrobike](https://www.capmetro.org/bikeshare) stations around the University of Texas at Austin campus and the surrounding West Campus area.

## The Model
The model consists of a number of stations where bikes can be picked up and dropped off, and a number of destinations that agents (commuters) want to visit. This is reprsented by a graph that agents can traverse.

For example, consider the following graph:

{{< figure src="./images/networkexample.png" width="250px" align="center" caption="Example graph of destinations (in blue) and stations (in red). The numbers on the nodes are the index of the node.">}}

In this graph, the blue nodes represent destinations and the red nodes represent stations. Every node is connected to every other node by an edge, whose weight represents the time it takes to travel between them on a bike (to get the time between two nodes by walking, multiply the edge weight by 3 since walking is about 3 times slower than biking). The fact that the graph is fully connected means that, in principle, an agent can travel between any two nodes by walking.

### Agent Decision Making
If an agent wants to bike, they must find a station with an available bike and another station where they can return the bike (ideally, the agent would want to find a station close to their destination to drop off the bike). Based on the fact that stations can be full or empty, the agent must decide whether to walk or bike to their destination. The full logic of the agent's pathfinding is quite tedious to explain, but trust that the agent will approximate your decision making to the first order. Interested readers can refer to the `pathfinding.py` file and the `step` method in the `Commuter` class in `commuter.py`.

## 
At each timestep, the agents will:
> 1. Move towards their destination
> 2. The stations update their bike counts based on the agents' decisions
> 3. Agents who arrived at their destination pick a new destination and start moving towards it
> 4. The way agents choose a particular destination is based on a probability distribution that we can set. 

Using the previous graph as an example, we can set the probability distribution to be uniform for every destination, or we can set it so that agents favor certain destinations over others. This is done with the `weights` attribute in the `MyModel` class in `model.py`.

## Optimization
The goal of the optimization is to find the best positions for the stations in the network. We can use two algorithms to do this: particle swarm optimization (PSO) and a genetic algorithm (GA). The optimization algorithms are implemented in the `optimize.py` file.

Essentially, we treat the position of all $n$ stations we want to optimize as a single vector $x \in \mathbb{R}^{2n}$, where each station has an $(x, y)$ coordinate. So, the optimization problem is to find the best $x$ that minimizes a fitness function. The fitness function we use is the negative of the average trips completed per agent, which we denote as $L$:
$$L = -\frac{T}{N}$$
where $T$ is the total number of trips completed by all agents and $N$ is the number of agents in the model. The reason we use the negative of the average trips completed is because the optimization algorithms are designed to minimize the fitness function, and we want to maximize the number of trips completed.

### PSO
Our implementation of PSO has the following hyperparameters:
- `n_particles`: The number of particles in the swarm.
- `n_iterations`: The number of iterations the algorithm will run for.
- `c1`: The cognitive parameter.
- `c2`: The social parameter.
- `w`: The inertia parameter.

The PSO algorithm works by initializing a swarm of particles with random positions and velocities. At each iteration, the particles update their positions and velocities based on their best position so far and the best position of the swarm. The best position of the swarm is the position that minimizes the fitness function. The particles then update their positions based on the following formula:
$$v_{i+1} = wv_i + c_1r_1(p_{\text{best}, i} - x_i) + c_2r_2(g_\text{best} - x_i)$$
$$x_{i+1} = x_i + v_{i+1}$$
where $v_i$ is the velocity of particle $i$, $x_i$ is the position of particle $i$, $p_{\text{best}, i}$ is the best position of particle $i$ so far, $g_{\text{best}}$ is the best position of the swarm, $r_1$ and $r_2$ are random numbers between 0 and 1, and $w$, $c_1$, and $c_2$ are the inertia, cognitive, and social parameters, respectively.

The PSO algorithm will do this for `n_iterations` iterations, and at the end, it will return the best position of the swarm found so far.

### Genetic Algorithm
Our implementation of the genetic algorithm has the following hyperparameters:
- `population_size`: The number of individuals in the population.
- `n_generations`: The number of generations the algorithm will run for.
- `mutation_rate`, $p$: The probability that a gene will mutate.
- `alpha`: The strength of the mutation.

Genetic algorithms works by initializing a population of individuals with random genes. At each generation, the individuals are evaluated based on their fitness, and the best individuals are selected to reproduce. The reproduction process involves selecting two parents and creating a child by combining their genes. The child's genes are then mutated with a certain probability. The best individuals from the previous generation are carried over to the next generation. The genetic algorithm will do this for `n_generations` generations, and at the end, it will return the best individual found so far.

In our case, the genes of an individual is a vector of $2n$ elements, where each pair of elements represents the $(x, y)$ coordinate of a station out of $n$ total stations. The fitness of an individual is the negative of the average trips completed per agent, as defined above. Given the two fittest individuals, $a$ and $b$, the child's genes are given by:
$$c_i = \left[\begin{cases}
    a_i  & \text{with probability } 0.5 \\
    b_i & \text{with probability } 0.5
\end{cases}\right] + 
\begin{cases}
    \alpha B & \text{with probability } p \\
    0 & \text{with probability } 1 - p
\end{cases}
$$
where $B$ is the length of the maxiumum dimension of the search space, $p$ is the mutation rate, and $\alpha$ is the strength of the mutation.

# Results
Everything needed to reproduce the results can be found in the `metrobike.ipynb` notebook. The notebook will guide you through the process of running the simulations and visualizing the results.

## Simple 2 station, 2 destination case
This is the simplest case we can consider. Here, the analytical solution is quite simple to find if we assume uniform weights for the destinations. The optimal positions for the station are to place them directly on top of the destinations, and both PSO and GA are able to find this solution quite easily:

{{< figure src="./images/pso_2_2.png" width="400px" align="center" caption="Optimized map of two destinations and two stations using the PSO algorithm.">}}
{{< figure src="./images/ga_2_2.png" width="400px" align="center" caption="Optimized map of two destinations and two stations using the GA algorithm.">}}

## 4 station, 2 destination case
For this case, we placed four destinations in a diamond shape. And destinations 2 and 1 are about twice as far away from each other than destination 3 and destination 4.

{{< figure src="./images/diamond.png" width="300px" align="center" caption="Map of four destinations arranged in a diamond shape. The aspect ratio of this graph is misleading, `Destination 2` and `Destination 1` are about twice as far away from each other than `Destination 4` and `Destination 3`">}}

Thus, with only two stations to place, the optimal solution is to place one station near destination 2 and the other near destination 1. Both the PSO and the genetic algorithm are quite sensitive to this problem, it seems as if they only converge to the optimal solution about half the time. For the PSO, we show a failed case, and for the GA we show a successful case where the optimal solution was found:

{{< figure src="./images/pso_4_2_uniform.png" width="400px" align="center" caption="Optimized map of four destinations and two stations using the PSO algorithm. Every destination is assumed to be equally popular.">}} 
{{< figure src="./images/ga_4_2_uniform.png" width="400px" align="center" caption="Optimized map of four destinations and two stations using the GA algorithm. Every destination is assumed to be equally popular">}}

Additionally, we can also consider the case where the weights are not uniform. For example, we can set the weights to be $(0.7, 0.1, 0.1, 0.1)$. In this case, the optimal solution is to place one station near destination 2 and the other near destination 1, since destination 1 will be the most popular. In this case, both PSO and GA are able to find the optimal solution:

{{< figure src="./images/pso_4_2_1weight.png" width="400px" align="center" caption="Optimized map of four destinations and two stations using the PSO algorithm. Station 1 was set to be the most popular with a probability of 0.7 while all other stations had a probability of 0.1.">}} 
{{< figure src="./images/ga_4_2_1weight.png" width="400px" align="center" caption="Optimized map of four destinations and two stations using the GA algorithm. Station 1 was set to be the most popular with a probability of 0.7 while all other stations had a probability of 0.1.">}}

## 4 station, 4 destination case
For this case, we placed four destinations in a square shape. The optimal solution is to place one station near each destination. However, both algorithms fail to find the optimal solution nearly every time and give us something like the following result:

{{< figure src="./images/pso_4_4.png" width="400px" align="center" caption="Optimized map of four destinations and two stations using the GA algorithm. All stations are equally popular.">}}

This is likely due to the fact that for this case, the optimal solution is hidden behind many local minima, and the algorithms are not able to escape them. It could be possible that more aggresive methods to jump out of local minima could help for this particular case of destinations=stations.

## Invariance to initial bike distribution
As one would expect, the initial distribution of bikes at the stations does not affect the final distribution of bikes at the stations. This is shown in the following histograms, where we start with all 10 bikes at station 1, but the distribution of bikes at the stations after 10,000 steps is equal (distribution collected for a model with 4 stations and 4 destinations, each with uniform weights):

{{< figure src="./images/uniformhist.PNG" width="600px" align="center" caption="Distribution of bikes held for each station. The map used was the example network shown in [The Model](#the-model) section with 4 destinations and 4 stations along rectangular vertices. Each destination was set to be equally popular.">}}

We can also alter the weights of the destinations to be $(0.7, 0.1, 0.1, 0.1)$ and observe how the invariant distribution is altered:

{{< figure src="./images/skewhist.PNG" width="600px" align="center" caption="Distribution of bikes held for each station. The map used was the example network shown in [The Model](#the-model). `Destination 1` was set to be the most pipular with a probability of 0.7 and the other destinations had a probability of 0.1.">}}

Even without seeing the actual map we used (we used the basic example graph shown in the very beginning), one can already guess that station 0 was placed closes to the most popular destination since it has the heaviest tail towards the right. From there, the agents seemed to prefer to bike to station 2 more often than station 3, and hardly any bikes ever reached station 1. So, despite station 1, 2, and 3 being just as popular of a destination, station 1 could had much less bikes than stations 2 or 3. 

Thus, we can see that the popularity of a destination is not the only factor that determines the distribution of bikes at nearby stations--we also need to consider the practicality riding a bike towards that destination from other nearby, popular destinations.

## Application to Real-World Data
Using metrobike data, we were able to estimate how popular certain areas in Austin were, and using GIS data of the distances between select locations of around campus and west campus, we were able to create a graph to represent UT Austin. 

{{< figure src="./images/destination_configuration.png" width="500px" align="center" caption="Map of select destinations around UT Austin and West Campus. The corresponding location to each node index and its estimated popularity is given in the [applications](#application-to-real-world-data) section.">}}

The select locations and their respective weights we used were (the list number corresponds to the destination number of the graph):
> 1. 26th West - 0.078
> 2. McCombs - 0.25
> 3. Target - 0.086
> 4. Union Building - 0.086
> 5. PMA - 0.14
> 6. Union on 24th - 0.071
> 7. Welch - 0.14
> 8. Rise - 0.021
> 9. Axis West - 0.077
> 10. Rec - 0.056

Then, we were able to use our optimization algorithms to find the optimal positions of the stations. We optimized for 6 stations around campus and west campus, and the results can be seen in the following images:


{{< figure src="./images/genetic10x6.png" width="400px" align="center" caption="Result of optimization of 6 station locations around UT and West Campus using GA optimization. Only four stations are visible since the GA failed to converge and placed two stations outside the plot bounds.">}}
{{< figure src="./images/pso10x6.png" width="400px" align="center" caption="Result of optimization of 6 station locations around UT and West Campus using PSO. The PSO algorithm converged to a fairly reasonable solution.">}}

We can see that the PSO algorithm was able to place all 6 stations within the boundaries we set, while the genetic algorithm placed 2 stations outside of the boundaries. In fact, the PSO solution seems very reasonable, with stations placed near the most popular destinations!

# Conclusion
For small systems, the optimization algorithms are able to find the optimal solution quite easily and quickly. For larger systems, however, we begin to see convergence issues. Nevertheless, given enough different initial conditions, the algorithms are able to find the optimal solution eventually--still faster than trying to brute force the solution as we allowed for a continious search of a $2n$ dimensional space, where $n$ is the number of stations whose locations we want to optimize.

We were able to find some interesting relationships between the popularity of a destination and the distribution of bikes at nearby stations. We also found that the initial distribution of bikes at the stations does not affect the final distribution of bikes at the stations, so the model has an invariant distribution of bikes that is reached rather quickly.

Finally, we were able to apply our model to real-world data and find the optimal positions of stations around campus and west campus. The results were quite reasonable, with stations placed near the most popular destinations.

# Future Work
Some directions we would really like to explore are
- Implementing a diurnal function to change the weights of destinations over time
- Implement a more agressive method to escape local minimas, which could help us find more optimal solutions for larger systems
- Visualizing the paths agents take to reach their destinations
- Do a grid search to find the best hyperparameters for the optimization algorithms, maybe this could also help stabilize convergence for larger systems