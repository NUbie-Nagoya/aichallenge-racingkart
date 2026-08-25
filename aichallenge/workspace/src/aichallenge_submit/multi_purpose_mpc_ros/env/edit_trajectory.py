import sys
import argparse
from os import path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.interpolate import splprep, splev

class TrajectoryEditor:
    def __init__(self, map_yaml_path, csv_path):
        self.vmax = 25.0 / 3.6  # Convert 25 km/h to m/s
        self.ay_max = 7.0       
        self.ax_max = 0.7       
        self.ax_min = -1.6      
        
        self.csv_path = csv_path
        
        # Load Map Data
        with open(map_yaml_path, 'r') as f:
            map_data = yaml.safe_load(f)
            
        base_path = path.dirname(map_yaml_path)
        image_array = np.array(mpimg.imread(path.join(base_path, map_data['image'])))
        
        res = map_data['resolution']
        origin = map_data['origin']
        size_x, size_y = image_array.shape[1], image_array.shape[0]
        
        extent = [origin[0], origin[0] + size_x * res, origin[1], origin[1] + size_y * res]
        
        # Setup Plot
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_title(f"Editing: {csv_path}\nLeft Click: Drag | Right Click: Remove | Press 's' to Save & Profiler")
        self.ax.imshow(np.flipud(image_array), cmap='gray', origin='lower', extent=extent)
        
        # Load existing trajectory to use as control points
        try:
            df = pd.read_csv(self.csv_path)
            # Thin out the points so it's actually editable (take every 20th point)
            self.wp_x = df['x_m'].iloc[::20].tolist()
            self.wp_y = df['y_m'].iloc[::20].tolist()
        except FileNotFoundError:
            print(f"❌ Could not find {self.csv_path}. Please run generate_track.py first.")
            sys.exit(1)
            
        self.dragging_idx = None
        self.df_out = None
        
        self.point_plot, = self.ax.plot(self.wp_x, self.wp_y, 'ro', markersize=6, picker=10, label="Control Points")
        self.spline_plot, = self.ax.plot([], [], 'b-', linewidth=2, label="Optimized Spline")
        self.ax.legend()
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.recalculate_trajectory()
        plt.show()

    def get_closest_point(self, x, y, threshold=2.0):
        distances = [np.hypot(x - px, y - py) for px, py in zip(self.wp_x, self.wp_y)]
        min_dist = min(distances)
        if min_dist < threshold:
            return distances.index(min_dist)
        return None

    def on_press(self, event):
        if event.inaxes != self.ax: return
        idx = self.get_closest_point(event.xdata, event.ydata)
        if event.button == 3 and idx is not None:
            self.wp_x.pop(idx)
            self.wp_y.pop(idx)
            self.update_plot()
        elif event.button == 1 and idx is not None:
            self.dragging_idx = idx

    def on_motion(self, event):
        if self.dragging_idx is None or event.inaxes != self.ax: return
        self.wp_x[self.dragging_idx] = event.xdata
        self.wp_y[self.dragging_idx] = event.ydata
        self.update_plot()

    def on_release(self, event):
        if event.button == 1:
            self.dragging_idx = None

    def on_key(self, event):
        if event.key == 's':
            self.save_trajectory()

    def update_plot(self):
        self.point_plot.set_data(self.wp_x, self.wp_y)
        if len(self.wp_x) >= 4:
            self.recalculate_trajectory()
        self.fig.canvas.draw_idle()

    def recalculate_trajectory(self):
        x = self.wp_x + [self.wp_x[0]]
        y = self.wp_y + [self.wp_y[0]]
        
        tck, u = splprep([x, y], s=0, per=True)
        u_new = np.linspace(0, 1.0, 1000)
        out = splev(u_new, tck)
        sx, sy = out[0], out[1]
        
        dx, dy = splev(u_new, tck, der=1)
        ddx, ddy = splev(u_new, tck, der=2)
        
        psi = np.arctan2(dy, dx)
        kappa = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-10)
        
        ds = np.sqrt(dx**2 + dy**2)
        du = u_new[1] - u_new[0]
        s_m = np.cumsum(ds * du)
        s_m -= s_m[0]
        
        v_lim = np.zeros_like(kappa)
        for i, k in enumerate(kappa):
            if abs(k) > 1e-5:
                v_lim[i] = min(np.sqrt(self.ay_max / abs(k)), self.vmax)
            else:
                v_lim[i] = self.vmax
                
        v_fwd = np.copy(v_lim)
        for i in range(1, len(v_fwd)):
            delta_s = s_m[i] - s_m[i-1]
            v_fwd[i] = min(v_fwd[i], np.sqrt(v_fwd[i-1]**2 + 2 * self.ax_max * delta_s))
            
        v_bwd = np.copy(v_fwd)
        for i in range(len(v_bwd)-2, -1, -1):
            delta_s = s_m[i+1] - s_m[i]
            v_bwd[i] = min(v_bwd[i], np.sqrt(v_bwd[i+1]**2 + 2 * abs(self.ax_min) * delta_s))
            
        vx_mps = v_bwd
        ax_mps2 = np.zeros_like(vx_mps)
        for i in range(1, len(vx_mps)-1):
            delta_s_total = s_m[i+1] - s_m[i-1]
            ax_mps2[i] = (vx_mps[i+1]**2 - vx_mps[i-1]**2) / (2 * delta_s_total)
            
        self.df_out = pd.DataFrame({
            's_m': s_m, 'x_m': sx, 'y_m': sy, 'psi_rad': psi,
            'kappa_radpm': kappa, 'vx_mps': vx_mps, 'ax_mps2': ax_mps2
        })
        self.spline_plot.set_data(sx, sy)

    def save_trajectory(self):
        if self.df_out is not None:
            self.df_out.to_csv(self.csv_path, index=False)
            print(f"✅ Trajectory & Velocities recalculated and saved to: {self.csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_yaml', type=str, required=True, help="Path to occupancy grid yaml")
    parser.add_argument('--csv', type=str, default="traj_mincurv.csv", help="CSV file to edit")
    args = parser.parse_args()
    
    TrajectoryEditor(args.map_yaml, args.csv)