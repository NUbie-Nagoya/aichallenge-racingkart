import csv
import math

def calculate_lap_time(csv_filepath):
    total_time = 0.0
    
    with open(csv_filepath, 'r') as file:
        # Assuming your CSV has headers like 'x', 'y', and 'v' (velocity in m/s)
        reader = csv.DictReader(file)
        rows = list(reader)
        
        for i in range(len(rows) - 1):
            x1, y1 = float(rows[i]['x_m']), float(rows[i]['y_m'])
            v1 = float(rows[i]['vx_mps'])
            
            x2, y2 = float(rows[i+1]['x_m']), float(rows[i+1]['y_m'])
            
            # Prevent division by zero if the kart is stopped
            if v1 <= 0.0:
                continue 
                
            # Delta t = Distance / Speed
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            time_segment = distance / v1
            
            total_time += time_segment
            
    return total_time

if __name__ == "__main__":
    # Replace with the actual path to your CSV file
    csv_file = "traj_mincurv_og.csv" 
    
    try:
        lap_time = calculate_lap_time(csv_file)
        print(f"🏎️  Theoretical Lap Time: {lap_time:.3f} seconds")
    except Exception as e:
        print(f"Error reading CSV: {e}\nCheck your column names (x, y, v).")