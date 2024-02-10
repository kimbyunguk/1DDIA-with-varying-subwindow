""" Visualizing wave propagation """
import numpy as np               
import matplotlib.pyplot as plt
from scipy.io            import loadmat, savemat

from matplotlib.animation import FuncAnimation, FFMpegWriter

TT, TL, dt, dx = int(15*60), 2300, 0.2, 0.5
t    =  np.arange(0,TT,dt)
x    =  np.arange(0,TL+dx,dx)

Case = ['M1', 'M2', 'M3', 'M4', 'M5']

fdir = 'Plae the location of input data'
savedir = 'Plae the location where to save the result'


for case in Case:    
    fdir = 'Plae your data location' + r'\{}\\'.format(Trial)
    
    Dataset = loadmat(fdir+'res_{}'.format(case))
    eta    = Dataset['eta']
    U      = Dataset['U']
    bathy  = Dataset['depth'].T[:eta.shape[1]]
    x      = Dataset['x'].T[:eta.shape[1]]
    t      = Dataset['t'].T
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20,3))
    
    # Create an empty plot with markers to update in the animation
    line, = ax.plot(x, bathy, '-', color='k')
    line, = ax.plot(x, eta[0,:], '-', color='r')
    
    # Set axis labels and title
    ax.xaxis.grid(True,linestyle='--', linewidth=0.3)
    ax.yaxis.grid(True,linestyle='--', linewidth=0.3)
    ax.set_xlabel('Cross-shore distance [m]')
    ax.set_ylabel('Elevation [m]')
    ax.set_title('Case: {}'.format(case))
    ax.set_ylim(np.floor(-np.max(abs(bathy)))-0.5,3)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    # Define the initialization function for the animation
    def init():
        line.set_ydata(np.ma.array(x, mask=True))  # mask the initial y data
        time_text.set_text('')
        return line, time_text
    
    # Define the update function for the animation
    def update(frame):
        y = eta[frame,:]  # update y values based on frame number
        line.set_ydata(y)  # update y data of the line plot
        time_text.set_text('time = {} sec'.format(int(frame*dt)))
        return line, time_text
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(eta), init_func=init, blit=True, interval = 1)
    
    output_file = r'A:\DIA\TWF\data\Res\Animation\{}.mp4'.format(case)
    
    # Use the FFMpegWriter to save the animation as an MP4 file
    # Make sure you have ffmpeg installed on your system
    writer = FFMpegWriter(fps=30)  # You can adjust the frames per second (fps)
    
    # Save the animation as an MP4 file
    plt.tight_layout()
    
    anim.save(output_file, writer=writer)
    
    plt.show()
    
    plt.close('all')
