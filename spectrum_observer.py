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

spectral.settings.envi_support_nonlowercase_params = 'TRUE'
# global variables
hsi: np.ndarray = None

# Create the main Tkinter window
root = tk.Tk()
root.title("Image Selection and Pixel Extraction")
root.resizable(0,0)
s = ttk.Style()
s.theme_use('clam')

def choose_cwd():
    '''Let user select directory where they import data'''

    _ = filedialog.askdirectory(initialdir= os.getcwd())
    if _:
        os.chdir(_)

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
        canvas.load_array(hsi)

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

    if os.path.exists(name+".png"):
        response = messagebox.askquestion("File Exists", "A file with the same name already exists. Do you want to overwrite?")
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

    image.save(name+".png")
    print("Image saved successfully.")

def update_legend():
    leg = ax.legend(fancybox=True, shadow=True)
    origlines = [value for value in plt_lines.values()]     # make a list of value in dict while preserving order (>Python3.7)
    for legline, origline in zip(leg.get_lines(), origlines):
        if legline not in [x for x in plt_lined.keys()]:    # avoid apply modifications to old Line2D objects more than one time
            legline.set_picker(True)  # Enable picking on the legend line.
            plt_lined[legline] = origline
            legline.set_linewidth(3)

def on_pick(event):
    legline = event.artist
    origline = plt_lined[legline]
    if event.mouseevent.button == 1:        # left mouse click
        # find the original line corresponding to the legend proxy line, and toggle its visibility.
        visible = not origline.get_visible()
        origline.set_visible(visible)
        # Change the alpha on the line in the legend, so we can see what lines have been toggled.
        legline.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()

    if event.mouseevent.button == 3:        # right mouse click
        origline.remove()       # remove the line from figure
        del plt_lines[origline.get_label()]
        update_legend()
        fig.canvas.draw()

def show_graph():

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

    try:
        name = os.path.basename(image_path).split('.')[0]
        indname = naming(name, [x for x in plt_lines.keys()])

    except:
        print('No existing file, please load file')
        return
    position = (np.asarray(canvas.mask) == canvas.mask_color)[:,:,3]    # extract a 2 dimensional boolean numpy array
    wavelengths = np.linspace(470, 900, num= hsi.shape[2])
    spectrum = []

    for _ in range(hsi.shape[2]):                                       # iterate over all bands of hsi
        masked_data = np.ma.array(hsi[:,:,_], mask= ~position)          # invert the mask since False means valid element in np.ma
        masked_mean = np.ma.mean(masked_data)
        spectrum.append(masked_mean)

    # create plot
    plt_lines[indname], = ax.plot(wavelengths, spectrum, color= canvas.graph_color, label= indname)

    update_legend()
    # Show the plot
    plt.show()

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

    def run(self):                                                                  # overwrites run() method from parent class
        extention_type = os.path.splitext(image_path)[1]
        if extention_type == '.npy':
            self.hsi_cache = np.load(image_path)
        else:
            arr = envi.open(image_path , image_path.replace('.hdr','.raw')).load()
            self.hsi_cache = np.asarray(arr)

class ZoomDrag(tk.Canvas):
    def __init__(self, master: any, width:int= 1000, height:int= 750, bg = 'black', **kwargs):
        super().__init__(master, width=width, height=height, bg= bg, **kwargs)
        self.scale_factor = tk.IntVar(value= 1)
        self.effective_offset = (0,0)
        self.arr = None
        self.image_id = None
        self.mask_id = None
        self.drawing = False
        self.lines = []
        self.mask_color = (255,0,0,40)
        self.graph_color:str = '#ff0000'

        self.bind('<ButtonPress-1>', self.start_drag)
        self.bind('<B1-Motion>', self.drag)
        self.bind('<ButtonRelease-1>', self.stop_drag)
        self.bind('<MouseWheel>', self.zoom)
        self.bind('<Button-4>', self.zoom)
        self.bind('<Button-5>', self.zoom)
        self.bind("<ButtonPress-3>", self.start_drawing)
        self.bind("<ButtonRelease-3>", self.stop_drawing)
        self.bind("<B3-Motion>", self.draw)
        self.bind('<Button-2>', self.pick_color)

    def load_array(self, arr):
        self.delete('all')
        self.arr = arr
        self.scale_factor.set(1)
        self.effective_offset = (0,0)
        self.image_id = None
        self.mask_id = None
        if arr is not None:
            onechan = arr[:,:,49]
            normed_gray = (((onechan - np.min(onechan)) / (np.max(onechan) - np.min(onechan))) * 255).astype('uint8')
            normed_gray = Image.fromarray(normed_gray, mode= 'L')
            self.pil_image = normed_gray
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
        width = int(self.pil_image.width * self.scale_factor.get())
        height = int(self.pil_image.height * self.scale_factor.get())

        self.resized_image = self.pil_image.resize((width, height), Image.Resampling.NEAREST)    # used nearest to avoid interpolation
        self.photo_image = ImageTk.PhotoImage(self.resized_image)

        anchor_x = int(self.winfo_width()/2)+self.effective_offset[0]*self.scale_factor.get()
        anchor_y = int(self.winfo_height()/2)+self.effective_offset[1]*self.scale_factor.get()
        if self.image_id:
            self.delete(self.image_id)
        self.image_id = self.create_image(anchor_x, anchor_y, image=self.photo_image, anchor='center', tags= ('image'))
        if self.mask_id:
            self.resized_mask = self.mask.resize((self.mask.width*self.scale_factor.get(),self.mask.height*self.scale_factor.get()),
                                                 Image.Resampling.NEAREST)
            self.tkmask = ImageTk.PhotoImage(self.resized_mask)
            self.mask_id = self.create_image(anchor_x, anchor_y, image=self.tkmask, anchor= 'center', tags= ('image'))

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
        self.mask_id = self.create_image(bbox[0], bbox[1], image=self.tkmask, anchor=tk.NW, tags= ('image'))

    def pick_color(self, event):
        color = colorchooser.askcolor(title="Pick a Color", parent= root)
        if color:
            current_color.configure(bg= color[1])
            self.graph_color = color[1]     # HEX color code

# define widgets
canvas = ZoomDrag(root)
menu_bar = tk.Menu(root)
scale_monitor = tk.Frame(root, bg="gray80", bd=2, relief="solid", highlightbackground= 'black')
scale_label = ttk.Label(scale_monitor, textvariable= canvas.scale_factor, background= 'gray80')
color_monitor = tk.Frame(root)
color_button = ttk.Button(color_monitor, text= 'change', command= lambda: canvas.pick_color(None))
current_color = tk.Label(color_monitor, width= 1, height= 1, background= canvas.graph_color)
graph_button = ttk.Button(root, text= 'apply graph', command= show_graph)

# Create a File menu
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Change CWD", command= choose_cwd)
file_menu.add_command(label="Open", command= open_hsi)
file_menu.add_command(label="Save Canvas", command= save_image)

# Add the File menu to the menu bar
menu_bar.add_cascade(label="File", menu=file_menu)

# Configure the root window to use the menu bar
root.config(menu=menu_bar)

# place wigets
scale_monitor.grid(row= 0, column= 0, padx= (5, 0))
tk.Label(scale_monitor, text= 'scale: ', bg= 'gray80').grid(row= 0, column= 0, sticky= 'E')
scale_label.grid(row= 0, column= 1, sticky= 'W')
color_monitor.grid(row= 1, column= 0, padx= (5, 0))
tk.Label(color_monitor, text= 'graph color').grid(row= 0, column= 0, sticky= 'S', columnspan= 2)
color_button.grid(row= 1, column= 0, columnspan= 2)
tk.Label(color_monitor, text= 'current:').grid(row= 2, column= 0, pady= 10)
current_color.grid(row= 2, column= 1, sticky= 'we')
graph_button.grid(row= 2, column= 0)

canvas.grid(row= 0, column= 1, rowspan= 3, padx= 5, pady= 5)

# Initialize plots globally
fig, ax = plt.subplots()
fig.canvas.mpl_connect('pick_event', on_pick)
# starting from Python 3.7, the built-in dict class also preserves the insertion order of elements
plt_lines = {}                  # key= label, value = original line
plt_lined = {}                  # Will map legend lines to original lines

# Customize the plot
ax.set_title('Spectrum Graph')
plt.xlabel('wavelength')
plt.ylabel('relative intensity')
plt.grid(True)

# Start the Tkinter event loop
root.mainloop()