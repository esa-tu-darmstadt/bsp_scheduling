import pickle
import matplotlib.pyplot as plt
from saga.utils.draw import draw_network, draw_task_graph, draw_gantt
import saga_bsp as bsp

def visualize_results(results_dir, sync_time):
    """Load and visualize the worst-case results from simulated annealing."""
    
    for pkl_file in results_dir.rglob("*.pkl"):
        print(f"Loading {pkl_file}")
        
        with open(pkl_file, 'rb') as f:
            simulation = pickle.load(f)
        
        # Get the worst-case scenario (final state)
        final_state = simulation.iterations[-1]
        worst_network = final_state.best_network
        worst_task_graph = final_state.best_task_graph
        worst_schedule = final_state.best_schedule
        worst_base_schedule = final_state.best_base_schedule

        experiment_name = f"{pkl_file.parent.name}_vs_{pkl_file.stem}"
        print(f"Worst-case energy for {experiment_name}: {final_state.best_energy:.3f}")
        
        # Visualize using SAGA's utilities
        draw_network(worst_network)
        plt.title(f"Worst Network - {experiment_name}")
        
        draw_task_graph(worst_task_graph)  
        plt.title(f"Worst Task Graph - {experiment_name}")
        
        # Recalculate the BSP schedule so we can visualize the BSP gantt chart
        bsp_hardware = bsp.BSPHardware(network=worst_network, sync_time=sync_time)
        bsp_scheduler = simulation.scheduler.bsp_scheduler
        worst_bsp_schedule = bsp_scheduler.schedule(bsp_hardware, worst_task_graph)
        bsp.draw_bsp_gantt(worst_bsp_schedule)
        plt.title(f"Worst BSP Schedule - {experiment_name}")
        
        # Draw the Gantt chart for the worst-case base schedule 
        draw_gantt(worst_base_schedule)
        plt.title(f"Worst Base Schedule - {experiment_name}")
        
    plt.show()