# postprocessing code to do crack tip tracking with phase field points cloud
# use Exodus from ./exodus.py to extract d and coord at tstep, and use different
# ways to `reconstract' a direct crack (or simple filter out the crack tip point),
# compute the crack tip velocity based on crack tip location

import re
import numpy as np
from exodus import Exodus


def dist_search(model: Exodus, initial_tip, d_name="d", d_c=0.95, tstep=0, block_id=0):
    """search to find crack tip at tstep given exodus obj, d, and dcr by diatace
    to the initial tip location, only works for single crack
    model: Exodus object
    initial_tip: location of initial crack tip, np.array([x0, y0])
    d_name: var name for phase field
    d_c: critical phase field value (tip will be searched in x(d > d_c))
    tstep: timestep
    block_id: block id of mesh
    return [tip_coord_x, tip_coord_y]
    """
    verts, _ = model.get_mesh(block_id, tstep)
    d = model.get_var(d_name, timestep=tstep)
    d_ind = np.where(d > d_c)[0]
    if np.size(d_ind) < 1:
        return initial_tip
    x = verts[d_ind, 0]
    y = verts[d_ind, 1]

    # only search for y>0 for now
    y = (np.abs(y) + y) / 2
    dist = (x - initial_tip[0]) ** 2 + (y - initial_tip[1]) ** 2
    tip_ind = np.where(dist == np.max(dist))
    return np.column_stack((x[tip_ind], y[tip_ind]))[0]

def crack_tip_tracking(model: Exodus, initial_tip, interval=1, d_name="d", d_c=0.95,
                       save_csv = False, save_dir = None):
    """loop over tsteps, extract crack tip location, calculate crack tip velocity
    model: Exodus object
    initial_tip: np.array([x0, y0])
    interval: time step interval, determine the resolution of crack tip verlocity
    output file: time, tip_coord_x, tip_coord_y, tip_velocity
    """
    # initialization
    time = model.get_time()

    # loop over time
    for i in range(0, np.size(time), interval):
        if(i==0):
            crack_tip_list = np.hstack((time[0], initial_tip, [0]))
        else:
            cur_tip_loc = dist_search(model, initial_tip, d_name, d_c, i)
    
            # calculate tip velocity
            dt = time[i] - time[i-1]
            if(i==1):
                old_tip_x, old_tip_y = crack_tip_list[1:3]
            else:
                old_tip_x, old_tip_y = crack_tip_list[i-1][1:3]
            tip_vel = np.sqrt((cur_tip_loc[0] - old_tip_x)**2 + (cur_tip_loc[1] - old_tip_y)**2)/dt
            cur_tip_info = np.hstack(([time[i]], cur_tip_loc, [tip_vel]))
            crack_tip_list = np.vstack((crack_tip_list, cur_tip_info))
    
    # save to csv if required
    if(save_csv is True):
        if(save_dir is None):
            save_dir = re.sub(r'\.[^.]+$', "_tip.csv", model.dir)
        np.savetxt(save_dir, crack_tip_list, delimiter=',', comments="", fmt="%.4e",
                   header="time,tip_x,tip_y,tip_vel")
    
    return crack_tip_list
