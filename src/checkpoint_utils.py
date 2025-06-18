import os
import pickle
from datetime import datetime
from typing import List, Tuple, Optional
import logfire

def list_checkpoints(checkpoint_dir: str = "checkpoints") -> List[Tuple[str, str, int, str]]:
    """
    List all available checkpoint files.
    
    Returns:
        List of tuples containing (filename, problem_name, step, timestamp)
    """
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith("_checkpoint.pkl"):
            filepath = os.path.join(checkpoint_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                problem_name = checkpoint_data.get('specification_name', 'Unknown')
                step = checkpoint_data.get('step', 0)
                timestamp = checkpoint_data.get('timestamp', 'Unknown')
                
                checkpoints.append((filename, problem_name, step, timestamp))
            except Exception as e:
                logfire.warning(f"Could not read checkpoint {filename}: {str(e)}")
    
    return sorted(checkpoints, key=lambda x: x[3], reverse=True)  # Sort by timestamp, newest first

def delete_checkpoint(problem_name: str, evolver_type: str = "evolver", checkpoint_dir: str = "checkpoints") -> bool:
    """
    Delete a specific checkpoint file.
    
    Args:
        problem_name: Name of the problem
        evolver_type: Type of evolver ("evolver" or "evolver2")
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        True if deleted successfully, False otherwise
    """
    suffix = "_evolver2_checkpoint.pkl" if evolver_type == "evolver2" else "_checkpoint.pkl"
    filename = f"{problem_name.replace(' ', '_')}{suffix}"
    filepath = os.path.join(checkpoint_dir, filename)
    
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            logfire.info(f"Deleted checkpoint: {filepath}")
            return True
        except Exception as e:
            logfire.error(f"Failed to delete checkpoint {filepath}: {str(e)}")
            return False
    else:
        logfire.warning(f"Checkpoint file not found: {filepath}")
        return False

def checkpoint_exists(problem_name: str, evolver_type: str = "evolver", checkpoint_dir: str = "checkpoints") -> bool:
    """
    Check if a checkpoint exists for a specific problem and evolver type.
    
    Args:
        problem_name: Name of the problem
        evolver_type: Type of evolver ("evolver" or "evolver2")
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        True if checkpoint exists, False otherwise
    """
    suffix = "_evolver2_checkpoint.pkl" if evolver_type == "evolver2" else "_checkpoint.pkl"
    filename = f"{problem_name.replace(' ', '_')}{suffix}"
    filepath = os.path.join(checkpoint_dir, filename)
    
    return os.path.exists(filepath)

def get_checkpoint_info(problem_name: str, evolver_type: str = "evolver", checkpoint_dir: str = "checkpoints") -> Optional[dict]:
    """
    Get information about a specific checkpoint.
    
    Args:
        problem_name: Name of the problem
        evolver_type: Type of evolver ("evolver" or "evolver2")
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Dictionary with checkpoint information or None if not found
    """
    suffix = "_evolver2_checkpoint.pkl" if evolver_type == "evolver2" else "_checkpoint.pkl"
    filename = f"{problem_name.replace(' ', '_')}{suffix}"
    filepath = os.path.join(checkpoint_dir, filename)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        return {
            'problem_name': checkpoint_data.get('specification_name', 'Unknown'),
            'step': checkpoint_data.get('step', 0),
            'timestamp': checkpoint_data.get('timestamp', 'Unknown'),
            'population_size': len(checkpoint_data['population'].get_population()) if 'population' in checkpoint_data else 0,
            'best_fitness': checkpoint_data['population'].get_best().evaluation.fitness if 'population' in checkpoint_data else None
        }
    except Exception as e:
        logfire.error(f"Failed to read checkpoint info: {str(e)}")
        return None

def print_checkpoint_status(checkpoint_dir: str = "checkpoints"):
    """Print a summary of all available checkpoints."""
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print("No checkpoints found.")
        return
    
    print(f"Found {len(checkpoints)} checkpoint(s):")
    print("-" * 80)
    print(f"{'Filename':<40} {'Problem':<20} {'Step':<6} {'Timestamp':<15}")
    print("-" * 80)
    
    for filename, problem_name, step, timestamp in checkpoints:
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%m/%d %H:%M")
        except:
            formatted_time = timestamp[:15]
            
        print(f"{filename:<40} {problem_name:<20} {step:<6} {formatted_time}")