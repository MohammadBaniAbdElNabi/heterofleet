"""
NSGA-III Multi-Objective Optimization for Task Allocation.

Implements the Non-dominated Sorting Genetic Algorithm III for
finding Pareto-optimal task allocations.

Based on Deb & Jain (2014) and HeteroFleet Architecture v1.0
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


@dataclass
class Individual:
    """Individual in the genetic algorithm population."""
    
    chromosome: np.ndarray = field(default_factory=lambda: np.array([]))
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    constraints: np.ndarray = field(default_factory=lambda: np.array([]))
    rank: int = 0
    crowding_distance: float = 0.0
    reference_point_idx: int = -1
    generation: int = 0
    evaluated: bool = False
    
    @property
    def total_constraint_violation(self) -> float:
        return float(np.sum(np.maximum(0, self.constraints)))
    
    @property
    def is_feasible(self) -> bool:
        return self.total_constraint_violation == 0
    
    def dominates(self, other: Individual) -> bool:
        """Check if this individual dominates another."""
        if self.is_feasible and not other.is_feasible:
            return True
        if not self.is_feasible and other.is_feasible:
            return False
        if not self.is_feasible and not other.is_feasible:
            return self.total_constraint_violation < other.total_constraint_violation
        
        at_least_as_good = np.all(self.objectives <= other.objectives)
        strictly_better = np.any(self.objectives < other.objectives)
        return at_least_as_good and strictly_better
    
    def copy(self) -> Individual:
        return Individual(
            chromosome=self.chromosome.copy(),
            objectives=self.objectives.copy() if len(self.objectives) > 0 else np.array([]),
            constraints=self.constraints.copy() if len(self.constraints) > 0 else np.array([]),
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            reference_point_idx=self.reference_point_idx,
            generation=self.generation,
            evaluated=self.evaluated,
        )


@dataclass
class ParetoFront:
    """Pareto front of non-dominated solutions."""
    
    individuals: List[Individual] = field(default_factory=list)
    generation: int = 0
    
    def add(self, individual: Individual) -> bool:
        for existing in self.individuals:
            if existing.dominates(individual):
                return False
        
        self.individuals = [
            ind for ind in self.individuals
            if not individual.dominates(ind)
        ]
        self.individuals.append(individual)
        return True
    
    @property
    def size(self) -> int:
        return len(self.individuals)
    
    def get_objective_matrix(self) -> np.ndarray:
        if not self.individuals:
            return np.array([])
        return np.array([ind.objectives for ind in self.individuals])
    
    def get_ideal_point(self) -> np.ndarray:
        if not self.individuals:
            return np.array([])
        return np.min(self.get_objective_matrix(), axis=0)


class ReferencePointGenerator:
    """Generates structured reference points for NSGA-III."""
    
    @staticmethod
    def generate_uniform(num_objectives: int, divisions: int) -> np.ndarray:
        def generate_recursive(left, total, depth, current):
            if depth == num_objectives - 1:
                current.append(left)
                return [current.copy()]
            
            points = []
            for i in range(left + 1):
                new_current = current + [i]
                points.extend(generate_recursive(left - i, total, depth + 1, new_current))
            return points
        
        points = generate_recursive(divisions, divisions, 0, [])
        return np.array(points) / divisions


class NSGA3Optimizer:
    """NSGA-III Multi-Objective Optimizer."""
    
    def __init__(
        self,
        num_objectives: int,
        num_variables: int,
        variable_bounds: List[Tuple[int, int]],
        population_size: int = 100,
        max_generations: int = 200,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        reference_divisions: int = 12
    ):
        self.num_objectives = num_objectives
        self.num_variables = num_variables
        self.variable_bounds = variable_bounds
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        self.reference_points = ReferencePointGenerator.generate_uniform(
            num_objectives, reference_divisions
        )
        self.population_size = max(population_size, len(self.reference_points))
        
        self._evaluate_func: Optional[Callable] = None
        self.population: List[Individual] = []
        self.pareto_front = ParetoFront()
        self.generation = 0
        self._stats = {"generations": [], "front_sizes": []}
    
    def set_objective_function(self, func: Callable) -> None:
        self._evaluate_func = func
    
    def _evaluate_individual(self, individual: Individual) -> None:
        if self._evaluate_func is None:
            raise ValueError("Objective function not set")
        
        objectives, constraints = self._evaluate_func(individual.chromosome)
        individual.objectives = np.array(objectives)
        individual.constraints = np.array(constraints) if len(constraints) > 0 else np.array([])
        individual.evaluated = True
    
    def _initialize_population(self) -> None:
        self.population = []
        for _ in range(self.population_size):
            chromosome = np.array([
                random.randint(bounds[0], bounds[1])
                for bounds in self.variable_bounds
            ])
            individual = Individual(chromosome=chromosome, generation=0)
            self._evaluate_individual(individual)
            self.population.append(individual)
    
    def _non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        n = len(population)
        dominated_by = [[] for _ in range(n)]
        domination_count = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                if population[i].dominates(population[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif population[j].dominates(population[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1
        
        fronts = []
        current_front = [i for i in range(n) if domination_count[i] == 0]
        
        while current_front:
            fronts.append([population[i] for i in current_front])
            for i in current_front:
                population[i].rank = len(fronts) - 1
            
            next_front = []
            for i in current_front:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        return fronts
    
    def _normalize_objectives(self, population: List[Individual]) -> np.ndarray:
        objectives = np.array([ind.objectives for ind in population])
        ideal = np.min(objectives, axis=0)
        translated = objectives - ideal
        
        intercepts = np.max(translated, axis=0)
        intercepts = np.maximum(intercepts, 1e-6)
        return translated / intercepts
    
    def _associate_to_reference_points(self, population: List[Individual], normalized: np.ndarray) -> None:
        for i, ind in enumerate(population):
            min_dist = float('inf')
            min_idx = 0
            
            for j, ref in enumerate(self.reference_points):
                ref_norm = np.linalg.norm(ref)
                if ref_norm > 1e-6:
                    proj_length = np.dot(normalized[i], ref) / (ref_norm ** 2)
                    proj = proj_length * ref
                    dist = np.linalg.norm(normalized[i] - proj)
                else:
                    dist = np.linalg.norm(normalized[i])
                
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
            
            ind.reference_point_idx = min_idx
            ind.crowding_distance = min_dist
    
    def _niching_selection(self, last_front: List[Individual], num_to_select: int, 
                          reference_counts: Dict[int, int]) -> List[Individual]:
        selected = []
        remaining = list(last_front)
        
        while len(selected) < num_to_select and remaining:
            ref_indices = list(set(ind.reference_point_idx for ind in remaining))
            min_count = min(reference_counts.get(idx, 0) for idx in ref_indices)
            min_refs = [idx for idx in ref_indices if reference_counts.get(idx, 0) == min_count]
            
            selected_ref = random.choice(min_refs)
            associated = [ind for ind in remaining if ind.reference_point_idx == selected_ref]
            
            if associated:
                if reference_counts.get(selected_ref, 0) == 0:
                    best = min(associated, key=lambda x: x.crowding_distance)
                else:
                    best = random.choice(associated)
                
                selected.append(best)
                remaining.remove(best)
                reference_counts[selected_ref] = reference_counts.get(selected_ref, 0) + 1
        
        return selected
    
    def _select_parents(self) -> List[Individual]:
        parents = []
        for _ in range(self.population_size):
            i, j = random.sample(range(len(self.population)), 2)
            ind_i, ind_j = self.population[i], self.population[j]
            
            if ind_i.rank < ind_j.rank:
                parents.append(ind_i)
            elif ind_j.rank < ind_i.rank:
                parents.append(ind_j)
            elif ind_i.crowding_distance > ind_j.crowding_distance:
                parents.append(ind_i)
            else:
                parents.append(ind_j)
        return parents
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, self.num_variables - 1)
        child1_chrom = np.concatenate([parent1.chromosome[:point], parent2.chromosome[point:]])
        child2_chrom = np.concatenate([parent2.chromosome[:point], parent1.chromosome[point:]])
        
        return (Individual(chromosome=child1_chrom, generation=self.generation + 1),
                Individual(chromosome=child2_chrom, generation=self.generation + 1))
    
    def _mutate(self, individual: Individual) -> Individual:
        chromosome = individual.chromosome.copy()
        for i in range(self.num_variables):
            if random.random() < self.mutation_prob:
                bounds = self.variable_bounds[i]
                chromosome[i] = random.randint(bounds[0], bounds[1])
        individual.chromosome = chromosome
        return individual
    
    def _create_offspring(self, parents: List[Individual]) -> List[Individual]:
        offspring = []
        random.shuffle(parents)
        
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = self._crossover(parents[i], parents[i + 1])
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            self._evaluate_individual(child1)
            self._evaluate_individual(child2)
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def _environmental_selection(self, combined: List[Individual]) -> List[Individual]:
        fronts = self._non_dominated_sort(combined)
        next_population = []
        front_idx = 0
        
        while len(next_population) + len(fronts[front_idx]) <= self.population_size:
            next_population.extend(fronts[front_idx])
            front_idx += 1
            if front_idx >= len(fronts):
                break
        
        if len(next_population) < self.population_size and front_idx < len(fronts):
            last_front = fronts[front_idx]
            num_remaining = self.population_size - len(next_population)
            
            all_selected = next_population + last_front
            normalized = self._normalize_objectives(all_selected)
            self._associate_to_reference_points(all_selected, normalized)
            
            reference_counts = {}
            for ind in next_population:
                idx = ind.reference_point_idx
                reference_counts[idx] = reference_counts.get(idx, 0) + 1
            
            selected = self._niching_selection(last_front, num_remaining, reference_counts)
            next_population.extend(selected)
        
        return next_population
    
    def _update_pareto_front(self) -> None:
        for ind in self.population:
            if ind.rank == 0:
                self.pareto_front.add(ind.copy())
        self.pareto_front.generation = self.generation
    
    def optimize(self, callback: Callable = None) -> ParetoFront:
        logger.info(f"Starting NSGA-III: {self.num_objectives} objectives, "
                   f"{self.num_variables} variables")
        
        self._initialize_population()
        self._non_dominated_sort(self.population)
        self._update_pareto_front()
        
        for gen in range(self.max_generations):
            self.generation = gen
            
            parents = self._select_parents()
            offspring = self._create_offspring(parents)
            combined = self.population + offspring
            self.population = self._environmental_selection(combined)
            self._update_pareto_front()
            
            self._stats["generations"].append(gen)
            self._stats["front_sizes"].append(self.pareto_front.size)
            
            if callback and not callback(gen, self.population, self.pareto_front):
                break
            
            if (gen + 1) % 20 == 0:
                logger.info(f"Generation {gen + 1}: Front size = {self.pareto_front.size}")
        
        return self.pareto_front
    
    def get_best_compromise(self, weights: np.ndarray = None) -> Individual:
        if self.pareto_front.size == 0:
            raise ValueError("Pareto front is empty")
        
        if weights is None:
            weights = np.ones(self.num_objectives) / self.num_objectives
        
        objectives = self.pareto_front.get_objective_matrix()
        ideal = np.min(objectives, axis=0)
        nadir = np.max(objectives, axis=0)
        
        range_obj = nadir - ideal
        range_obj[range_obj == 0] = 1.0
        normalized = (objectives - ideal) / range_obj
        
        scores = np.sum(normalized * weights, axis=1)
        return self.pareto_front.individuals[np.argmin(scores)]
