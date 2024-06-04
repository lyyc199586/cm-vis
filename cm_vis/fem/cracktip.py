# postprocessing code to do crack tip tracking with phase field points cloud
# use Exodus from ./exodus.py to extract d and coord at tstep, and use different
# ways to `reconstract' a direct crack (or simple filter out the crack tip point),
# compute the crack tip velocity based on crack tip location

# TODO: not accurate in tip finding! and very slow use paraview contours instead for now

import re
import csv
import numpy as np
from cm_vis.fem.exodus import Exodus


def dist_search(model: Exodus, tip_init, tip_old, d_name="d", d_c=0.95, dist_old=0, tstep=0, block_id=0):
    """search to find crack tip at tstep given exodus obj, d, and dcr by diatace
    to the initial tip location, only works for single crack
    model: Exodus object
    tip_init: location of initial crack tip, np.array([x0, y0])
    tip_old: location of old crack tip, np.array([tip_x_old, tip_y_old])
    d_name: var name for phase field
    d_c: critical phase field value (tip will be searched in x(d > d_c))
    tstep: timestep
    block_id: block id of mesh
    return [tip_x, tip_y]
    """
    verts, _ = model.get_mesh(block_id, tstep)
    d = model.get_var(d_name, timestep=tstep)
    d_ind = np.where((d > d_c))[0]
    if np.size(d_ind) < 1:
        return tip_init
    x = verts[d_ind, 0]
    y = verts[d_ind, 1]

    # only search for y>0 for now
    y = np.abs(y)
    dist = (x - tip_init[0]) ** 2 + (y - tip_init[1]) ** 2
    valid_ind = np.where((dist >= dist_old) & (x >= tip_old[0]) & (y >= tip_old[1]))[0]
    max_dist = np.max(dist[valid_ind])
    tip_ind = np.where(dist == max_dist)[0]

    return np.column_stack((x[tip_ind], y[tip_ind]))[0]

def crack_tip_tracking(model: Exodus, tip_init, interval=1, d_name="d", d_c=0.95,
                       save_csv = False, save_dir = None):
    """loop over time steps, extract crack tip location, calculate crack tip velocity
    model: Exodus object
    tip_init: np.array([x0, y0])
    interval: time step interval, determine the resolution of crack tip velocity
    output file: time, tip_coord_x, tip_coord_y, tip_velocity
    """
    # initialization
    time = model.get_time()
    dist_old = 0
    
    # open csv if required
    if save_csv:
        if save_dir is None:
            save_dir = re.sub(r'\.[^.]+$', "_tip.csv", model.dir)
        with open(save_dir, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["time", "tip_x", "tip_y", "tip_vel"])
            print(f"CSV file opened and header written to {save_dir}")
        
    
    # loop over time
    for i in range(0, np.size(time), interval):
        if(i==0):
            crack_tip_list = np.hstack((time[0], tip_init, [0]))
            tip_old = tip_init
            print("Current crack tip info:\n Time, coord_x, coord_y, velocity")
            if save_csv:
                with open(save_dir, mode='a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(crack_tip_list)
            else:
                print(crack_tip_list)
        else:
    
            # calculate tip velocity
            dt = time[i] - time[i-1]
            if(i==1):
                tip_x_old, tip_y_old = crack_tip_list[1:3]
            else:
                tip_x_old, tip_y_old = crack_tip_list[i-1][1:3]
            tip_old = np.array([tip_x_old, tip_y_old])
            
            dist_old = (tip_x_old - tip_init[0]) ** 2 + (tip_y_old - tip_init[1]) ** 2
            cur_tip_loc = dist_search(model, tip_init, tip_old, d_name, d_c, dist_old, i)
            tip_vel = np.sqrt((cur_tip_loc[0] - tip_x_old)**2 + (cur_tip_loc[1] - tip_y_old)**2)/dt
            cur_tip_info = np.hstack(([time[i]], cur_tip_loc, [tip_vel]))
            print(cur_tip_info)
            crack_tip_list = np.vstack((crack_tip_list, cur_tip_info))
            if save_csv:
                with open(save_dir, mode='a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([f'{num:.4e}' for num in cur_tip_info])
    
    return crack_tip_list
