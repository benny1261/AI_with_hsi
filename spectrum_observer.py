import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, ttk
from spectral.io import envi
import spectral
from PIL import ImageTk, Image, ImageDraw
import numpy as np
import os
import re
from scipy.spatial import ConvexHull
from threading import Thread
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import csv
import cv2
from collections.abc import MutableSequence

spectral.settings.envi_support_nonlowercase_params = 'TRUE'
# global variables
hsi: np.ndarray = None
destination:str = os.getcwd()
des_flag:bool = False
fig3d_flag:bool = False
fig_flag:bool = False

# Create the main Tkinter window
root = tk.Tk()
root.title("Image Selection and Pixel Extraction")
root.resizable(0,0)
s = ttk.Style()
s.theme_use('clam')

def choose_cwd():
    '''Let user select directory where they import data'''
    global destination

    _ = filedialog.askdirectory(initialdir= os.getcwd())
    if _:
        os.chdir(_)
        if not des_flag:
            destination = _

def change_des():
    '''Let user select directory where they export data'''
    global destination, des_flag

    _ = filedialog.askdirectory(initialdir= destination)
    if _:
        destination = _
        des_flag = True

def open_hsi():
    global image_path, tic
    image_path = filedialog.askopenfilename(filetypes=[("Array",'*.npy;*.hdr')],initialdir= os.getcwd())   

    if image_path:
        import_td = Import_thread()
        progwin = TLProgressBar(root)
        progwin.title('Importing')

        tic = time.time()
        import_td.start()           # start the run() method in Import_thread
        thread_monitor(progwin, import_td)

def thread_monitor(window, thread):
    '''check whether the thread in window is alive, if not, run command'''
    global hsi

    if thread.is_alive():
        window.after(100, lambda: thread_monitor(window, thread))
    else:
        # pass data after thread closed
        window.destroy()
        hsi = thread.hsi_cache
        toc = time.time()
        print('loading time:', toc-tic, 'seconds')
        statusbar.configure(text= f'{os.path.basename(image_path)} (h:{hsi.shape[0]},w:{hsi.shape[1]},c:{hsi.shape[2]})')
        canvas.load_array(hsi, thread.normed_arr_cache)

        # enable some functions
        hough_button.configure(state= 'normal')

def longest_slice_between_duplicate_elements(lst):
    seen = {}       # Dictionary to store the seen elements and their indices
    longest_slice = []

    for i, element in enumerate(lst):
        if element in seen:
            current_slice = lst[seen[element]:i]  # Get the slice between the duplicates
            if len(current_slice) > len(longest_slice):
                longest_slice = current_slice

        seen[element] = i       # Store the element and its index
    return longest_slice        # return empty array if no same element found

def save_image():
    '''saves the current image in canvas window'''
    try:
        name = os.path.basename(image_path).split('.')[0]
    except:
        return

    if os.path.exists(os.path.join(destination, name) + ".png"):
        response = messagebox.askquestion("File Exists", "A file with same name already exists. Confirm overwrite?")
        if response == 'no':
            return

    image:Image.Image = canvas.resized_image
    bbox = canvas.bbox(canvas.image_id)
    x1 = max(-bbox[0], 0)
    y1 = max(-bbox[1], 0)
    x2 = x1+ min(image.width, canvas.winfo_width())
    y2 = y1+ min(image.height, canvas.winfo_height())
    image = image.crop((x1,y1,x2,y2))

    if canvas.mask_id:
        mask:Image.Image = canvas.resized_mask
        mask = mask.crop((x1,y1,x2,y2))
        image = image.convert('RGBA')
        image = Image.alpha_composite(image, mask)

    image.save(os.path.join(destination, name) + ".png")
    print("Image saved successfully to", destination)

def show_graph():
    '''show 2d line graph of the area(mean) across spectral dimension'''
    global plot2d
    try:
        name = os.path.basename(image_path).split('.')[0]
    except:
        print('No existing file, please load file')
        return
    
    if not hasattr(canvas, 'mask'):
        print('No selected area')
        return

    if not fig_flag:
        plot2d = PlotSpectrum()
    plot2d.add_data(name)
    plot2d.update_legend()
    # Show the plot
    plot2d.fig.show()

def on_closing():
    plt.close('all')
    root.destroy()

def show_3d():
    '''show 3d plot on one layer across spatial dimension'''
    global plot3d
    # block errors
    try:
        image_path
    except:
        print('No existing file, please load file')
        return
    if not hasattr(canvas, 'mask'):
        print('No selected area')
        return

    if not fig3d_flag:
        plot3d = PlotSlice()

    plot3d.update_data()
    plot3d.update_band()
    plot3d.fig3d.show()

def export_data():
    '''export csv file'''
    csv_path:str = os.path.join(destination, 'data.csv')
    if os.path.exists(csv_path):
        response = messagebox.askquestion("File Exists", "A file with same name already exists. Confirm overwrite?")
        if response == 'no':
            return

    data= {}
    first_iter:bool= True
    for key, value in plot2d.plt_lines.items():
        if first_iter:
            wavelengths = value.get_xdata()
            first_iter= False
        data[key]= value.get_ydata()

    # Save the dictionary to a CSV file
    with open(csv_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Wavelengths", *wavelengths])
        for key, value in data.items():
            csvwriter.writerow([key, *value])

    print("Data saved successfully to", destination)

def switch_frame(*event):
    slaves = root.grid_slaves(0,0)
    if slaves:
        slaves[0].grid_remove()
    frame= frames[mode.get()]
    frame.grid(row= 0, column= 0, padx= (PADX, 0), sticky= 'NSEW')

    # switch settings in canvas
    canvas.switch_mode()

class PlotSpectrum:
    def __init__(self) -> None:
        global fig_flag
        # Initialize 2Dplots
        self.fig = plt.figure('spectral plot')
        self.ax = self.fig.add_subplot(111)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick_legend)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        # starting from Python 3.7, the built-in dict class also preserves the insertion order of elements
        self.plt_lines = {}                 # key= label, value = original line
        self.plt_lined = {}                 # Will map legend lines to original lines
        # Customize the plot
        self.ax.set_title('Spectrum Graph')
        self.ax.set_xlabel('wavelength')
        self.ax.set_ylabel('relative intensity')
        self.ax.grid(True)

        fig_flag = True

    def update_legend(self):
        leg = self.ax.legend(fancybox=True, shadow=True)
        leg.set_draggable(True)
        origlines = [value for value in self.plt_lines.values()]    # make a list of value in dict while preserving order (>Python3.7)
        for legline, origline in zip(leg.get_lines(), origlines):
            if legline not in [x for x in self.plt_lined.keys()]:   # avoid apply modifications to old Line2D objects more than one time
                legline.set_picker(True)    # Enable picking on the legend line.
                self.plt_lined[legline] = origline
                legline.set_linewidth(3)
        
        # update export availability
        if len(self.plt_lines) > 0:
            save_spec_button.configure(state= 'normal')
        else:
            save_spec_button.configure(state= 'disabled')

    def on_pick_legend(self, event):
        legline = event.artist
        if legline not in [x for x in self.plt_lined.keys()]:   # blocking mouse event from other object when set_draggable set to True
            return
        origline = self.plt_lined[legline]
        if event.mouseevent.button == 1:        # left mouse click
            # find the original line corresponding to the legend proxy line, and toggle its visibility.
            visible = not origline.get_visible()
            origline.set_visible(visible)
            # Change the alpha on the line in the legend, so we can see what lines have been toggled.
            legline.set_alpha(1.0 if visible else 0.2)
            self.fig.canvas.draw()

        if event.mouseevent.button == 3:        # right mouse click
            origline.remove()       # remove the line from figure
            del self.plt_lines[origline.get_label()]
            self.update_legend()
            self.fig.canvas.draw()

    def add_data(self, name):

        def naming(substr:str, list_to_search:list[str])->str:
            '''avoid identical names'''

            matching_names = [n for n in list_to_search if n.startswith(substr)]
            pattern = r'.*\((\d+)\)$'
            index_cache = []

            if matching_names:
                for matching_name in matching_names:
                    match = re.search(pattern, matching_name)
                    if match:       # indexed name exists
                        index_cache.append(int(match.group(1)))

                if not index_cache: return substr+'(1)'
                else:
                    return substr+f'({max(index_cache)+1})'

            return substr

        indname = naming(name, [x for x in self.plt_lines.keys()])
        position = (np.asarray(canvas.mask) == canvas.mask_color)[:,:,3]    # extract a 2 dimensional boolean numpy array
        wavelengths = np.linspace(470, 900, num= hsi.shape[2])
        spectrum = []

        for _ in range(hsi.shape[2]):                                       # iterate over all bands of hsi
            masked_data = np.ma.array(hsi[:,:,_], mask= ~position)          # invert the mask since False means valid element in np.ma
            masked_mean = np.ma.mean(masked_data)
            spectrum.append(masked_mean)

        # create plot and keep reference of it
        self.plt_lines[indname], = self.ax.plot(wavelengths, spectrum, color= canvas.graph_color, label= indname)

    def on_close(self, event):
        global fig_flag
        fig_flag = False
        save_spec_button.configure(state= 'disabled')

class PlotSlice:
    def __init__(self) -> None:
        global fig3d_flag
        self.index:int = 49
        self.bbox_hsi:np.ndarray = None
        self.bbox_mask:np.ndarray = None

        # Initialize a 3D subplot
        self.fig3d = plt.figure('spatial plot')
        self.fig3d.canvas.mpl_connect('key_press_event', self.update_band)
        self.fig3d.canvas.mpl_connect('close_event', self.on_close)
        self.ax3d = self.fig3d.add_subplot(111, projection= '3d')
        self.cbar = self.fig3d.colorbar(cm.ScalarMappable(norm= None, cmap= cm.coolwarm), ax= self.ax3d)

        fig3d_flag = True

    def update_data(self):
        def mask_bbox(boolmask:np.ndarray)->tuple:
            rows = np.any(boolmask, axis= 1)
            cols = np.any(boolmask, axis= 0)
            rmin, rmax = np.where(rows)[0][[0, -1]]     # np.where returns: (array of index which is True, dtype=int64)
            cmin, cmax = np.where(cols)[0][[0, -1]]     # advanced indexing [[0,-1]] to extract first and last element in 1D array
            return rmin, rmax+1, cmin, cmax+1           # +1 is for index(exclude in end of slicing)

        position = (np.asarray(canvas.mask) == canvas.mask_color)[:,:,3]    # extract a 2 dimensional boolean numpy array
        bbox = mask_bbox(position)

        self.bbox_hsi = hsi[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        self.bbox_mask = position[bbox[0]:bbox[1],bbox[2]:bbox[3]]

        # Create a grid for X and Y coordinates
        X = np.arange(0, self.bbox_mask.shape[1], 1)
        Y = np.arange(0, self.bbox_mask.shape[0], 1)
        self.X, self.Y = np.meshgrid(X, Y)

    def update_band(self, event= None):
        '''update band and make plot on axis[0] in figure'''

        if event is not None:
            if event.key == 'z':                # must be aware it can only read english keys
                self.index = max(0, self.index-1)
            elif event.key == 'c':
                self.index = min(self.bbox_hsi.shape[2]-1, self.index+1)

        # Clear existing plots in ax3d from previous calls
        self.ax3d.cla()

        bbox_slice = self.bbox_hsi[:,:,self.index]
        bbox_slice_masked = np.where(self.bbox_mask == True, bbox_slice, np.nan)

        # Customize the z axis.
        self.ax3d.set_zlim(np.nanmin(bbox_slice_masked), np.nanmax(bbox_slice_masked))     # uses nanmin/nanmax to ignore np.nan values
        self.ax3d.zaxis.set_major_locator(LinearLocator(10))
        self.ax3d.zaxis.set_major_formatter('{x:.02f}')

        # update color bar limits
        self.cbar.mappable.set_clim(vmin = np.nanmin(bbox_slice_masked), vmax = np.nanmax(bbox_slice_masked))

        # Plot the surface.
        self.ax3d.plot_surface(self.X, self.Y, bbox_slice_masked, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # update show index
        self.ax3d.set_title(f'\nband {self.index+1}')

        self.fig3d.canvas.draw()

    def on_close(self, event):
        global fig3d_flag
        fig3d_flag = False

class TLProgressBar(tk.Toplevel):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.master = master
        self.resizable(0,0)
        self.attributes('-topmost', 'true')

        # progress bar widget
        progbar = ttk.Progressbar(self, mode= "indeterminate", length= 500)
        progbar.start()
        progbar.pack(padx= 20, pady= 10)

        # initializing bar position
        self.update_idletasks()             # update information of toplevel size after placing widget
        cord_x = self.master.winfo_x()+(self.master.winfo_width()-self.winfo_width())/2
        cord_y = self.master.winfo_y()+(self.master.winfo_height()-self.winfo_height())/2
        self.geometry(f'+{int(cord_x)}+{int(cord_y)}')

class Import_thread(Thread):                                                        # define a class that inherits from 'Thread' class
    def __init__(self):
        super().__init__()                                                          # run __init__ of parent class
        self.hsi_cache= None
        self.normed_arr_cache= None

    def run(self):                                                                  # overwrites run() method from parent class
        extention_type = os.path.splitext(image_path)[1]
        if extention_type == '.npy':
            self.hsi_cache = np.load(image_path)
        else:
            arr = envi.open(image_path , image_path.replace('.hdr','.raw')).load()
            self.hsi_cache = np.asarray(arr)
        self.normed_arr_cache = self.norm_arr(self.hsi_cache)

    def norm_arr(self, arr:np.ndarray)->np.ndarray:
        stack_list=[]
        for channel in range(arr.shape[2]):
            onechan = arr[:,:,channel]
            # normed_gray = (((onechan - np.min(onechan)) / (np.max(onechan) - np.min(onechan))) * 255).astype('uint8')
            normed_gray = cv2.normalize(onechan, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            stack_list.append(normed_gray)
        return np.stack(stack_list, axis=2)

class CallbackList(MutableSequence):
    def __init__(self, *args):
        self._list = list(args)
        self._callback = None

    def set_callback(self, callback):
        self._callback = callback

    def _trigger_callback(self):
        if self._callback:
            self._callback(len(self))

    def __len__(self):
        return len(self._list)

    def __getitem__(self, index):
        return self._list[index]

    def __setitem__(self, index, value):
        self._list[index] = value
        self._trigger_callback()

    def __delitem__(self, index):
        del self._list[index]
        self._trigger_callback()

    def insert(self, index, value):
        self._list.insert(index, value)
        self._trigger_callback()

    def __repr__(self):
        return repr(self._list)

class ZoomDrag(tk.Canvas):
    def __init__(self, master: any, width:int= 1000, height:int= 750, bg = 'black', **kwargs):
        super().__init__(master, width=width, height=height, bg= bg, **kwargs)
        self.scale_factor = tk.IntVar(value= 1)
        self.band = tk.IntVar(value= 50)
        self.dots = CallbackList()
        self.dots_len = tk.IntVar(value= 0)
        self.dots.set_callback(lambda x:self.dots_len.set(x))
        self.init_attribute()
        self.drawing = False
        self.lines = []         # will clear itself after every draw
        self.mask_color = (255,0,0,40)
        self.graph_color:str = '#ff0000'

        self.bind('<ButtonPress-1>', self.start_drag)
        self.bind('<B1-Motion>', self.drag)
        self.bind('<ButtonRelease-1>', self.stop_drag)
        self.bind('<MouseWheel>', self.zoom)
        self.bind('<Button-4>', self.zoom)
        self.bind('<Button-5>', self.zoom)
        self.analysis_binds()

    def init_attribute(self):
        self.delete('all')
        self.arr = None
        self.normed_arr = None
        self.scale_factor.set(1)
        self.effective_offset = (0,0)
        self.image_id = None
        self.mask_id = None
        self.dots.clear()

    def switch_mode(self):
        # bindings
        self.unbind_dynamic()
        if mode.get() == 'analysis':
            self.analysis_binds()
        if mode.get() == 'hough':
            self.hough_binds()
        # images
        self.itemconfigure('dynamic', state= 'hidden')
        self.itemconfigure(mode.get(), state= 'normal')

    def analysis_binds(self):
        self.bind("<ButtonPress-3>", self.start_drawing)
        self.bind("<ButtonRelease-3>", self.stop_drawing)
        self.bind("<B3-Motion>", self.draw)
        self.bind('<Button-2>', self.pick_color)

    def hough_binds(self):
        pass

    def unbind_dynamic(self):
        self.unbind("<ButtonPress-3>")
        self.unbind("<ButtonRelease-3>")
        self.unbind("<B3-Motion>")
        self.unbind('<Button-2>')

    def load_array(self, arr, normed_arr):
        self.init_attribute()
        self.arr = arr
        self.normed_arr = normed_arr
        self.update_band()
        if hasattr(self, 'mask'):
            del self.mask

            graph_button.configure(state= 'disabled')
            plot3d_button.configure(state= 'disabled')

    def update_band(self, *event):
        if np.any(self.arr) == None:
            return
        self.normed_gray = self.normed_arr[:,:,self.band.get()-1]
        # xx = cv2.HoughCircles(
        #     self.normed_gray,
        #     cv2.HOUGH_GRADIENT_ALT,
        #     dp=1.5,
        #     minDist=6,
        #     param1=10,
        #     param2=0.5,
        #     minRadius=6,
        #     maxRadius=16)
        self.pil_image = Image.fromarray(self.normed_gray, mode= 'L')
        self.update_image()
        self.addtag("image", 'withtag', self.image_id)

    def start_drag(self, event):
        self.start_x, self.start_y = event.x, event.y
    
    def drag(self, event):
        if self.drawing or not self.image_id:
            return
        dx = event.x- self.start_x
        dy = event.y- self.start_y
        self.move('image', dx, dy)

        # constrain movement
        bbox = self.bbox(self.image_id)
        if (bbox[2]-bbox[0])<= self.winfo_width():
            if bbox[0] < 0:
                self.move('image', -bbox[0], 0)
            if bbox[2] > self.winfo_width():
                self.move('image', self.winfo_width() - bbox[2], 0)
        else:
            if bbox[0] > 0:
                self.move('image', -bbox[0], 0)
            if bbox[2] < self.winfo_width():
                self.move('image', self.winfo_width() - bbox[2], 0)

        if (bbox[3]-bbox[1])<= self.winfo_height():
            if bbox[1] < 0:
                self.move('image', 0, -bbox[1])
            if bbox[3] > self.winfo_height():
                self.move('image', 0, self.winfo_height() - bbox[3])
        else:
            if bbox[1] > 0:
                self.move('image', 0, -bbox[1])
            if bbox[3] < self.winfo_height():
                self.move('image', 0, self.winfo_height() - bbox[3])
        # self.configure(scrollregion=self.bbox(self.image_id))     # makes the item in canvas unable to exceed border of canvas
                                                                    # but occers error in coordinate when drag over border

        self.start_x, self.start_y = event.x, event.y

    def stop_drag(self, event):
        if self.drawing or not self.image_id:
            return

        # update offset based on true position
        bbox = self.bbox(self.image_id)
        offset_x = (bbox[0]+bbox[2]- self.winfo_width())// 2
        offset_y = (bbox[1]+bbox[3]- self.winfo_height())// 2
        self.effective_offset = offset_x//self.scale_factor.get(), offset_y//self.scale_factor.get()

    def update_image(self):
        '''show pil_image based on its scale and position, mask included if exist'''
        width = int(self.pil_image.width * self.scale_factor.get())
        height = int(self.pil_image.height * self.scale_factor.get())

        self.resized_image = self.pil_image.resize((width, height), Image.Resampling.NEAREST)    # used nearest to avoid interpolation
        self.photo_image = ImageTk.PhotoImage(self.resized_image)

        anchor_x = int(self.winfo_width()/2)+self.effective_offset[0]*self.scale_factor.get()
        anchor_y = int(self.winfo_height()/2)+self.effective_offset[1]*self.scale_factor.get()
        if self.image_id:
            self.delete(self.image_id)
        self.image_id = self.create_image(anchor_x, anchor_y, image=self.photo_image, anchor='center', tags= ('image'))
        # analysis
        if self.mask_id:
            self.delete(self.mask_id)
            self.resized_mask = self.mask.resize((self.mask.width*self.scale_factor.get(),self.mask.height*self.scale_factor.get()),
                                                 Image.Resampling.NEAREST)
            self.tkmask = ImageTk.PhotoImage(self.resized_mask)
            self.mask_id = self.create_image(anchor_x, anchor_y, image=self.tkmask, anchor= 'center', tags= ('image','dynamic','analysis'))
            if mode.get() != 'analysis':
                self.itemconfigure('analysis', state= 'hidden')
        # hough
        if self.dots:
            bbox = self.bbox(self.image_id)
            for dot in self.dots:
                self.coords(dot.id, bbox[0]+dot.x*self.scale_factor.get(),bbox[1]+dot.y*self.scale_factor.get())
            if mode.get() != 'hough':
                self.itemconfigure('hough', state= 'hidden')

        self.tag_lower(self.image_id)   # move image to bottom of all masks

    def zoom(self, event):
        if self.drawing or not self.image_id:
            return

        if event.delta > 0 or event.num == 4:
            self.scale_factor.set(self.scale_factor.get()+1)
        elif event.delta < 0 or event.num == 5:
            self.scale_factor.set(max(self.scale_factor.get()-1, 1))

        self.update_image()

    def start_drawing(self, event):
        global draw_start_x, draw_start_y
        if not self.image_id:
            return
        self.drawing = True
        if self.mask_id:
            self.delete(self.mask_id)
        draw_start_x, draw_start_y = event.x, event.y

    def stop_drawing(self, event):
        global draw_start_x, draw_start_y
        if self.drawing:
            self.lines.append(self.create_line(draw_start_x, draw_start_y, event.x, event.y))
            self.fill_closed_shape()            
            for line in self.lines:
                self.delete(line)
            self.lines.clear()
            self.drawing = False

    def draw(self, event):
        global draw_start_x, draw_start_y
        if self.drawing:
            self.lines.append(self.create_line(draw_start_x, draw_start_y, event.x, event.y))
            draw_start_x, draw_start_y = event.x, event.y

    def fill_closed_shape(self):
        '''creates an alpha mask and show in canvas'''
        if len(self.lines)<2:
            return

        line_coords = []
        for line in self.lines:
            coords = self.coords(line)
            line_coords.append(tuple(coords[:2]))       # extract start point

        # capture largest closing if possible
        largest_slice = longest_slice_between_duplicate_elements(line_coords)
        if not largest_slice:
            largest_slice = line_coords

        # apply convex hull algorithm
        hull = ConvexHull(largest_slice)
        convex_coords = [largest_slice[i] for i in hull.vertices]

        # crop coordinates if exceed bbox of image
        bbox = self.bbox(self.image_id)
        canvas_coords = [(max(bbox[0], min(x, bbox[2])), max(bbox[1], min(y, bbox[3]))) for x, y in convex_coords]

        # translate coordinates with reference to image item
        item_coords = [(x-bbox[0], y-bbox[1]) for x, y in canvas_coords]

        self.mask = Image.new("RGBA", (bbox[2]-bbox[0], bbox[3]-bbox[1]), (0, 0, 0, 0))
        draw = ImageDraw.Draw(self.mask)
        draw.polygon(item_coords, fill= self.mask_color)

        if self.scale_factor.get() != 1:
            self.mask = self.mask.resize((self.pil_image.width, self.pil_image.height), Image.Resampling.NEAREST)       # 1xsize
            self.resized_mask = self.mask.resize((self.mask.width*self.scale_factor.get(),self.mask.height*self.scale_factor.get()),
                                                Image.Resampling.NEAREST)
            self.tkmask = ImageTk.PhotoImage(self.resized_mask)
        else:
            self.resized_mask = self.mask
            self.tkmask = ImageTk.PhotoImage(self.resized_mask)
        self.mask_id = self.create_image(bbox[0], bbox[1], image=self.tkmask, anchor=tk.NW, tags= ('image','dynamic','analysis'))

        # enable plots
        graph_button.configure(state= 'normal')
        plot3d_button.configure(state= 'normal')

    def pick_color(self, event):
        color = colorchooser.askcolor(title="Pick a Color", parent= root)
        if color:
            current_color.configure(bg= color[1])
            self.graph_color = color[1]     # HEX color code

    def calculate_hough(self, dp:float, mindist:float, param1:float, param2:float, minrad:int, maxrad:int):
        circles = cv2.HoughCircles(
            self.normed_gray,
            cv2.HOUGH_GRADIENT_ALT,
            dp= dp,
            minDist= mindist,
            param1= param1,
            param2= param2,
            minRadius= minrad,
            maxRadius= maxrad
        )

        rad:int = 3
        size:int= 7
        patch = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(patch)
        draw.ellipse((size//2-rad, size//2-rad, size//2+rad, size//2+rad), (0,0,255,100))
        self.tkpatch = ImageTk.PhotoImage(patch)

        bbox = self.bbox(self.image_id)
        # circles.shape (1, N, 3)
        if circles is None: return
        circles = np.squeeze(circles)
        circles = np.uint16(np.around(circles))
        for x, y, _rad in circles:
            dot = self.PatchManager(x, y)
            dot.id = self.create_image(bbox[0]+x, bbox[1]+y, image=self.tkpatch, anchor=tk.CENTER, tags= ('image','dynamic','hough'))
            self.dots.append(dot)

    class PatchManager:
        def __init__(self, x, y) -> None:
            self.x, self.y = x, y
            self.id = None

        def del_instance(self):
            del self

class CustomEntry(ttk.Entry):
    def __init__(self, master, root, allow_float:bool= False):
        validate_input_cmd = root.register(self.validate_input)
        super().__init__(master, validate="key", validatecommand=(validate_input_cmd, "%d", "%P")
                        , width= 10)
        self.allow_float = allow_float

    def validate_input(self, action, new_value:str):
        if action == '1':  # Insert action
            if self.allow_float:
                new_value = new_value.replace('.','',1) # only first decimal point is valid
            if new_value.isdigit():
                return True
            else:
                return False
        return True

# Closing protocal
root.protocol('WM_DELETE_WINDOW', on_closing)

# frame dict
frames={}

# define widgets
canvas = ZoomDrag(root)
menu_bar = tk.Menu(root)
statusbar = ttk.Label(root, text= 'Author C.C.Hsu 2023', border= 1, relief= 'sunken', anchor= 'e')

# Create menus
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Change CWD", command= choose_cwd)
file_menu.add_command(label="Open", command= open_hsi)
save_menu = tk.Menu(menu_bar, tearoff=0)
save_menu.add_command(label= "Export dir", command= change_des)
save_menu.add_command(label="Save Canvas", command= save_image)

# Add the File menu to the menu bar
menu_bar.add_cascade(label="File", menu=file_menu)
menu_bar.add_cascade(label="Save", menu=save_menu)

# Configure the root window to use the menu bar
root.config(menu=menu_bar)

# analysis widgets
frames['analysis'] = tk.Frame(root)
scale_monitor = tk.Frame(frames['analysis'], bg="gray80", bd=2, relief="solid", highlightbackground= 'black')
color_monitor = tk.Frame(frames['analysis'])
color_button = ttk.Button(color_monitor, text= 'change', command= lambda: canvas.pick_color(None))
current_color = tk.Label(color_monitor, width= 1, height= 1, background= canvas.graph_color)
graph2d_monitor = tk.Frame(frames['analysis'])
graph_button = ttk.Button(graph2d_monitor, text= 'apply graph', command= show_graph, state= 'disabled')
save_spec_button = ttk.Button(graph2d_monitor, text= 'save as csv', command= export_data, state= 'disabled')
plot3d_button = ttk.Button(frames['analysis'], text= '3D graph', command= show_3d, state= 'disabled')
band_monitor = tk.Frame(frames['analysis'])
band_label = tk.Label(band_monitor, text= canvas.band.get())        # didn't use textvariable cause ttk.Label will show lots of digits
band_scalebar = ttk.Scale(band_monitor, from_= 1, to= 150, variable= canvas.band, 
                          command= lambda x: (canvas.update_band(None), band_label.configure(text=canvas.band.get())))

# hough widgets
frames['hough'] = tk.Frame(root)
scale_monitor_h = tk.Frame(frames['hough'], bg="gray80", bd=2, relief="solid", highlightbackground= 'black')
hough_para_monitor = ttk.Frame(frames['hough'], border= 6)
dp_entry = CustomEntry(hough_para_monitor, root, True)
dp_entry.insert(0, 1.5)
mindist_entry = CustomEntry(hough_para_monitor, root, True)
mindist_entry.insert(0, 6)
canny_thres_entry = CustomEntry(hough_para_monitor, root, True)
canny_thres_entry.insert(0, 10)
roundness_thres_entry = CustomEntry(hough_para_monitor, root, True)
roundness_thres_entry.insert(0, 0.5)
minrad_entry = CustomEntry(hough_para_monitor, root)
minrad_entry.insert(0, 6)
maxrad_entry = CustomEntry(hough_para_monitor, root)
maxrad_entry.insert(0, 16)
hough_button = ttk.Button(frames['hough'], text= 'calculate', state= 'disabled', command= lambda: canvas.calculate_hough(
    float(dp_entry.get()), float(mindist_entry.get()), float(canny_thres_entry.get()), float(roundness_thres_entry.get()),
    int(minrad_entry.get()), int(maxrad_entry.get())))
amount_monitor = tk.Frame(frames['hough'])

# place widgets
PADX = 10
# analysis part
scale_monitor.grid(row= 0)
ttk.Label(scale_monitor, text= 'scale: ', background= 'gray80', foreground= 'blue',
          font= ('Times New Roman', 16)).grid(row= 0, column= 0, sticky= 'E')
ttk.Label(scale_monitor, textvariable= canvas.scale_factor, background= 'gray80', foreground= 'blue',
          font=('Times New Roman', 16)).grid(row= 0, column= 1, sticky= 'W')
color_monitor.grid(row= 1)
tk.Label(color_monitor, text= 'graph color').grid(row= 0, column= 0, sticky= 'S', columnspan= 2)
color_button.grid(row= 1, column= 0, columnspan= 2)
tk.Label(color_monitor, text= 'current:').grid(row= 2, column= 0, pady= 10)
current_color.grid(row= 2, column= 1, sticky= 'we')
graph2d_monitor.grid(row= 2)
graph_button.grid(row= 0, column= 0)
save_spec_button.grid(row= 1, column= 0, pady= (5,0))
plot3d_button.grid(row= 3, column= 0)
band_monitor.grid(row= 4)
tk.Label(band_monitor, text= 'band: ').grid(row= 0, column= 0, sticky='E')
band_label.grid(row= 0, column= 1, sticky='W')
band_scalebar.grid(row= 1, columnspan= 2)
# hough part
scale_monitor_h.grid(row= 0)
ttk.Label(scale_monitor_h, text= 'scale: ', background= 'gray80', foreground= 'blue',
          font= ('Times New Roman', 16)).grid(row= 0, column= 0, sticky= 'E')
ttk.Label(scale_monitor_h, textvariable= canvas.scale_factor, background= 'gray80', foreground= 'blue',
          font=('Times New Roman', 16)).grid(row= 0, column= 1, sticky= 'W')
entry_pady = (10, 0)
hough_para_monitor.grid(row=1)
ttk.Label(hough_para_monitor, text= 'Parameters', font=('Times New Roman', 16)).grid(row= 0, columnspan=2)
ttk.Label(hough_para_monitor, text= 'dp').grid(row= 1, column= 0, sticky= "W", pady= entry_pady)
dp_entry.grid(row= 1, column= 1, pady= entry_pady)
ttk.Label(hough_para_monitor, text= 'min dist').grid(row= 2, column= 0, sticky= "W", pady=entry_pady)
mindist_entry.grid(row= 2, column= 1, pady= entry_pady)
ttk.Label(hough_para_monitor, text= 'canny').grid(row= 3, column= 0, sticky= "W", pady=entry_pady)
canny_thres_entry.grid(row= 3, column= 1, pady= entry_pady)
ttk.Label(hough_para_monitor, text= 'roundness').grid(row= 4, column= 0, sticky= "W", pady=entry_pady)
roundness_thres_entry.grid(row= 4, column= 1, pady= entry_pady)
ttk.Label(hough_para_monitor, text= 'min rad').grid(row= 5, column= 0, sticky= "W", pady=entry_pady)
minrad_entry.grid(row= 5, column= 1, pady= entry_pady)
ttk.Label(hough_para_monitor, text= 'max rad').grid(row= 6, column= 0, sticky= "W", pady=entry_pady)
maxrad_entry.grid(row= 6, column= 1, pady= entry_pady)
amount_monitor.grid(row=2)
tk.Label(amount_monitor, text= 'amount:').grid(row=0, column= 0, sticky= "E")
tk.Label(amount_monitor, textvariable= canvas.dots_len).grid(row= 0, column= 1, sticky= 'W')
hough_button.grid(row=3)

# mode switch
mode = tk.StringVar(root, value= 'analysis')
mode_switch = ttk.OptionMenu(root, mode, 'analysis', *frames.keys(), command= switch_frame)
mode_switch.grid(row= 1, column= 0, padx= (PADX, 0), pady= 5)

statusbar.grid(row= 2, column=0, columnspan= 2, sticky= 'WE')
canvas.grid(row= 0, column= 1, rowspan= 2, padx= PADX, pady= 5)

# make widgets distribute equally
for frame in frames.values():
    frame.grid_columnconfigure([i for i in range(frame.grid_size()[0])], weight=1)
    frame.grid_rowconfigure([i for i in range(frame.grid_size()[1])], weight=1)
switch_frame()  # initialize call
root.grid_rowconfigure(0, weight= 1)

# Start the Tkinter event loop
root.mainloop()