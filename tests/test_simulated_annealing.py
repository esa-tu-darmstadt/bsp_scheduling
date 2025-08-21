#!/usr/bin/env python3
"""Comprehensive tests for BSP simulated annealing optimization"""

import unittest
import networkx as nx
from src.saga_bsp.schedule import BSPSchedule, BSPHardware
from src.saga_bsp.optimization import (
    BSPSimulatedAnnealing, 
    MoveTaskToSuperstep, 
    MoveTaskToProcessor, 
    DuplicateAndMoveTask
)


class TestBSPSimulatedAnnealing(unittest.TestCase):
    """Test cases for BSP simulated annealing optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.hardware = self.create_test_hardware()
        self.task_graph = self.create_test_task_graph()
    
    def create_test_hardware(self):
        """Create test hardware configuration"""
        network = nx.Graph()
        network.add_node("P0", weight=1.0)  # Slow processor
        network.add_node("P1", weight=3.0)  # Fast processor  
        network.add_node("P2", weight=2.0)  # Medium processor
        
        network.add_edge("P0", "P1", weight=10.0)
        network.add_edge("P1", "P2", weight=8.0)
        network.add_edge("P0", "P2", weight=6.0)
        
        return BSPHardware(network=network, sync_time=0.5)
    
    def create_test_task_graph(self):
        """Create test task dependency graph"""
        task_graph = nx.DiGraph()
        
        # Diamond pattern with parallel paths
        tasks = {
            "T1": 20.0,   # Entry task
            "T2": 10.0,   # Left branch
            "T3": 15.0,   # Right branch
            "T4": 25.0,   # Join task
            "T5": 12.0    # Exit task
        }
        
        for task, weight in tasks.items():
            task_graph.add_node(task, weight=weight)
        
        dependencies = [
            ("T1", "T2", 2.0),
            ("T1", "T3", 3.0),
            ("T2", "T4", 4.0),
            ("T3", "T4", 5.0),
            ("T4", "T5", 2.0)
        ]
        
        for src, dst, weight in dependencies:
            task_graph.add_edge(src, dst, weight=weight)
        
        return task_graph
    
    def create_suboptimal_schedule(self):
        """Create a deliberately suboptimal schedule"""
        schedule = BSPSchedule(self.hardware, self.task_graph)
        
        # Put everything on slow processor in separate supersteps
        ss0 = schedule.add_superstep()
        schedule.schedule("T1", "P0", ss0)
        
        ss1 = schedule.add_superstep()
        schedule.schedule("T2", "P0", ss1)
        
        ss2 = schedule.add_superstep()
        schedule.schedule("T3", "P0", ss2)
        
        ss3 = schedule.add_superstep()
        schedule.schedule("T4", "P0", ss3)
        
        ss4 = schedule.add_superstep()
        schedule.schedule("T5", "P0", ss4)
        
        return schedule
    
    def test_schedule_validation_valid(self):
        """Test that a valid schedule passes validation"""
        schedule = self.create_suboptimal_schedule()
        is_valid, errors = schedule.is_valid()
        
        self.assertTrue(is_valid, f"Valid schedule failed validation: {errors}")
        self.assertEqual(len(errors), 0)
    
    def test_schedule_validation_missing_task(self):
        """Test validation catches missing tasks"""
        schedule = BSPSchedule(self.hardware, self.task_graph)
        
        # Only schedule some tasks
        ss0 = schedule.add_superstep()
        schedule.schedule("T1", "P0", ss0)
        schedule.schedule("T2", "P0", ss0)
        
        is_valid, errors = schedule.is_valid()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("missing" in error.lower() for error in errors))
    
    def test_schedule_validation_precedence_violation(self):
        """Test validation catches precedence violations"""
        schedule = BSPSchedule(self.hardware, self.task_graph)
        
        # Create precedence violation: T4 before T2
        ss0 = schedule.add_superstep()
        schedule.schedule("T1", "P0", ss0)
        
        ss1 = schedule.add_superstep()
        schedule.schedule("T4", "P0", ss1)  # T4 depends on T2, but T2 is later
        
        ss2 = schedule.add_superstep()
        schedule.schedule("T2", "P0", ss2)
        schedule.schedule("T3", "P0", ss2)
        
        ss3 = schedule.add_superstep()
        schedule.schedule("T5", "P0", ss3)
        
        is_valid, errors = schedule.is_valid()
        
        self.assertFalse(is_valid)
        self.assertTrue(any("precedence violation" in error.lower() for error in errors))
    
    def test_move_task_to_superstep_feasibility(self):
        """Test MoveTaskToSuperstep feasibility checking"""
        schedule = self.create_suboptimal_schedule()
        
        # Test valid move: T2 from superstep 1 to superstep 2
        action = MoveTaskToSuperstep(2)
        self.assertTrue(action.is_feasible(schedule, "T2"))
        
        # Test invalid move: T4 to superstep 0 (would violate precedence)
        action = MoveTaskToSuperstep(0)
        self.assertFalse(action.is_feasible(schedule, "T4"))
    
    def test_move_task_to_processor_feasibility(self):
        """Test MoveTaskToProcessor feasibility checking with intra-superstep dependencies"""
        schedule = BSPSchedule(self.hardware, self.task_graph)
        
        # Create scenario with intra-superstep dependencies
        ss0 = schedule.add_superstep()
        schedule.schedule("T1", "P0", ss0)
        
        ss1 = schedule.add_superstep()
        schedule.schedule("T2", "P1", ss1)
        schedule.schedule("T4", "P1", ss1)  # T4 depends on T2, same superstep, same processor
        
        ss2 = schedule.add_superstep()
        schedule.schedule("T3", "P2", ss2)
        schedule.schedule("T5", "P2", ss2)
        
        # T4 cannot be moved to different processor because it depends on T2 in same superstep
        action = MoveTaskToProcessor("P2")
        self.assertFalse(action.is_feasible(schedule, "T4"))
        
        # T2 can be moved because it has no intra-superstep dependencies
        self.assertTrue(action.is_feasible(schedule, "T2"))
    
    def test_duplicate_and_move_task_action(self):
        """Test DuplicateAndMoveTask action"""
        schedule = BSPSchedule(self.hardware, self.task_graph)
        
        # Create scenario with intra-superstep dependencies
        ss0 = schedule.add_superstep()
        schedule.schedule("T1", "P0", ss0)
        
        ss1 = schedule.add_superstep()
        schedule.schedule("T2", "P1", ss1)
        schedule.schedule("T4", "P1", ss1)  # T4 depends on T2
        
        ss2 = schedule.add_superstep()
        schedule.schedule("T3", "P2", ss2)
        schedule.schedule("T5", "P2", ss2)
        
        initial_makespan = schedule.makespan
        
        # DuplicateAndMoveTask should be able to handle this
        action = DuplicateAndMoveTask("P2")
        self.assertTrue(action.is_feasible(schedule, "T4"))
        
        # Apply the action
        success = action.apply(schedule, "T4")
        self.assertTrue(success)
        
        # Verify schedule is still valid
        is_valid, errors = schedule.is_valid()
        self.assertTrue(is_valid, f"Schedule invalid after duplication: {errors}")
        
        # T2 should now be duplicated on P2
        t2_tasks = [task for task in schedule.task_mapping.values() if task.node == "T2"]
        processors_with_t2 = set(task.proc for task in t2_tasks)
        self.assertIn("P1", processors_with_t2)  # Original
        self.assertIn("P2", processors_with_t2)  # Duplicate
    
    def test_simulated_annealing_finds_improvement(self):
        """Test that simulated annealing finds better solutions"""
        initial_schedule = self.create_suboptimal_schedule()
        initial_makespan = initial_schedule.makespan
        
        # Verify initial schedule is valid
        is_valid, errors = initial_schedule.is_valid()
        self.assertTrue(is_valid, f"Initial schedule invalid: {errors}")
        
        # Run optimization
        optimizer = BSPSimulatedAnnealing(
            max_iterations=50,
            max_temp=10.0,
            min_temp=0.1,
            cooling_rate=0.9
        )
        
        optimized_schedule = optimizer.optimize(initial_schedule)
        optimized_makespan = optimized_schedule.makespan
        
        # Verify optimized schedule is valid
        is_valid, errors = optimized_schedule.is_valid()
        self.assertTrue(is_valid, f"Optimized schedule invalid: {errors}")
        
        # Should find improvement (or at least not get worse)
        self.assertLessEqual(optimized_makespan, initial_makespan, 
                           "Optimization should not increase makespan")
        
        # For this deliberately bad schedule, we expect significant improvement
        improvement_percent = (initial_makespan - optimized_makespan) / initial_makespan * 100
        self.assertGreater(improvement_percent, 10.0, 
                         f"Expected >10% improvement, got {improvement_percent:.1f}%")
    
    def test_simulated_annealing_with_all_actions(self):
        """Test simulated annealing with all action types"""
        initial_schedule = self.create_suboptimal_schedule()
        initial_makespan = initial_schedule.makespan
        
        # Use all action types
        optimizer = BSPSimulatedAnnealing(
            max_iterations=100,
            max_temp=20.0,
            min_temp=0.01,
            cooling_rate=0.95,
            action_types=[MoveTaskToSuperstep, MoveTaskToProcessor, DuplicateAndMoveTask]
        )
        
        optimized_schedule = optimizer.optimize(initial_schedule)
        optimized_makespan = optimized_schedule.makespan
        
        # Verify validity
        is_valid, errors = optimized_schedule.is_valid()
        self.assertTrue(is_valid, f"Optimized schedule invalid: {errors}")
        
        # Check statistics
        stats = optimizer.get_optimization_stats()
        
        self.assertGreater(stats['total_iterations'], 0)
        self.assertGreaterEqual(stats['acceptance_rate'], 0)
        self.assertLessEqual(stats['acceptance_rate'], 1)
        self.assertEqual(stats['initial_energy'], initial_makespan)
        self.assertEqual(stats['final_energy'], optimized_makespan)
        
        # Should find improvement
        self.assertLess(optimized_makespan, initial_makespan)
    
    def test_complex_task_graph_optimization(self):
        """Test optimization on a more complex task graph"""
        # Create complex task graph with multiple parallel paths
        complex_graph = nx.DiGraph()
        
        # Add nodes
        tasks = {f"T{i}": float(10 + i*5) for i in range(1, 11)}
        for task, weight in tasks.items():
            complex_graph.add_node(task, weight=weight)
        
        # Create complex dependencies
        edges = [
            ("T1", "T2", 1.0), ("T1", "T3", 1.0),
            ("T2", "T4", 2.0), ("T2", "T5", 2.0),
            ("T3", "T6", 2.0), ("T3", "T7", 2.0),
            ("T4", "T8", 3.0), ("T5", "T8", 3.0),
            ("T6", "T9", 3.0), ("T7", "T9", 3.0),
            ("T8", "T10", 4.0), ("T9", "T10", 4.0)
        ]
        
        for src, dst, weight in edges:
            complex_graph.add_edge(src, dst, weight=weight)
        
        # Create bad initial schedule
        schedule = BSPSchedule(self.hardware, complex_graph)
        for i, task in enumerate(sorted(complex_graph.nodes()), 1):
            ss = schedule.add_superstep() if i == 1 else schedule.supersteps[-1] if i % 2 == 0 else schedule.add_superstep()
            schedule.schedule(task, "P0", ss)  # All on slow processor
        
        initial_makespan = schedule.makespan
        
        # Optimize
        optimizer = BSPSimulatedAnnealing(
            max_iterations=200,
            max_temp=50.0,
            min_temp=0.1,
            cooling_rate=0.96
        )
        
        optimized_schedule = optimizer.optimize(schedule)
        
        # Verify
        is_valid, errors = optimized_schedule.is_valid()
        self.assertTrue(is_valid, f"Complex optimized schedule invalid: {errors}")
        
        improvement = (initial_makespan - optimized_schedule.makespan) / initial_makespan * 100
        self.assertGreater(improvement, 5.0, f"Expected >5% improvement on complex graph, got {improvement:.1f}%")


class TestScheduleActions(unittest.TestCase):
    """Test individual schedule actions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.hardware = self.create_test_hardware()
        self.task_graph = self.create_test_task_graph()
        
    def create_test_hardware(self):
        """Create test hardware configuration"""
        network = nx.Graph()
        network.add_node("P0", weight=1.0)
        network.add_node("P1", weight=2.0)
        
        network.add_edge("P0", "P1", weight=10.0)
        
        return BSPHardware(network=network, sync_time=1.0)
    
    def create_test_task_graph(self):
        """Create simple test task graph"""
        task_graph = nx.DiGraph()
        
        task_graph.add_node("A", weight=10.0)
        task_graph.add_node("B", weight=15.0)
        task_graph.add_node("C", weight=20.0)
        
        task_graph.add_edge("A", "B", weight=5.0)
        task_graph.add_edge("B", "C", weight=3.0)
        
        return task_graph
    
    def test_move_task_to_superstep_get_possible_targets(self):
        """Test getting possible superstep targets"""
        schedule = BSPSchedule(self.hardware, self.task_graph)
        
        ss0 = schedule.add_superstep()
        schedule.schedule("A", "P0", ss0)
        
        ss1 = schedule.add_superstep()
        schedule.schedule("B", "P0", ss1)
        
        ss2 = schedule.add_superstep()
        schedule.schedule("C", "P0", ss2)
        
        action = MoveTaskToSuperstep(0)  # Dummy
        targets = action.get_possible_targets(schedule, "B")
        
        # B can move to superstep 2 (after A, before C)
        self.assertIn(2, targets)
        # B cannot move to superstep 0 (before A) or stay in superstep 1
        self.assertNotIn(0, targets)
        self.assertNotIn(1, targets)
    
    def test_move_task_to_processor_get_possible_targets(self):
        """Test getting possible processor targets"""
        schedule = BSPSchedule(self.hardware, self.task_graph)
        
        ss0 = schedule.add_superstep()
        schedule.schedule("A", "P0", ss0)
        
        action = MoveTaskToProcessor("")  # Dummy
        targets = action.get_possible_targets(schedule, "A")
        
        # A can move to P1 but not P0 (already there)
        self.assertEqual(targets, ["P1"])
    
    def test_duplicate_and_move_comprehensive(self):
        """Comprehensive test of duplicate and move action"""
        schedule = BSPSchedule(self.hardware, self.task_graph)
        
        # Set up intra-superstep dependency
        ss0 = schedule.add_superstep()
        schedule.schedule("A", "P0", ss0)
        schedule.schedule("B", "P0", ss0)  # B depends on A, same superstep
        
        ss1 = schedule.add_superstep()
        schedule.schedule("C", "P1", ss1)
        
        # Move B to P1 should duplicate A
        action = DuplicateAndMoveTask("P1")
        success = action.apply(schedule, "B")
        
        self.assertTrue(success)
        
        # A should now exist on both processors
        a_tasks = [task for task in schedule.task_mapping.values() if task.node == "A"]
        a_processors = set(task.proc for task in a_tasks)
        self.assertEqual(len(a_processors), 2)
        self.assertIn("P0", a_processors)
        self.assertIn("P1", a_processors)
        
        # Verify schedule validity
        is_valid, errors = schedule.is_valid()
        self.assertTrue(is_valid, f"Schedule after duplication invalid: {errors}")


if __name__ == '__main__':
    unittest.main(verbosity=2)