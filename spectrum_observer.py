import tkinter as tk
from tkinter import filedialog
from spectral.io import envi
import spectral
from PIL import ImageTk, Image, ImageDraw
import numpy as np
import os
from scipy.spatial import ConvexHull

spectral.settings.envi_support_nonlowercase_params = 'TRUE'

# Create the main Tkinter window
window = tk.Tk()
window.title("Image Selection and Pixel Extraction")
window.resizable(0,0)

def open_hsi():
    global image_path, hsi
    image_path = filedialog.askopenfilename(filetypes=[("Array",'*.npy;*.hdr')],
                                            initialdir= r'D:\code\AI_with_hsi\data\slices')
    if image_path:
        extention_type = os.path.splitext(image_path)[1]
        if extention_type == '.npy':
            hsi = np.load(image_path)
        else:
            arr = envi.open(image_path , image_path.replace('.hdr','.raw')).load()
            hsi = arr.copy()
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


class ZoomDrag(tk.Canvas):
    def __init__(self, master: any, size: int = 400, bg = 'black', **kwargs):
        super().__init__(master, width=size, height=size, bg= bg, **kwargs)
        self.canv_len = size
        self.scale_factor:int  = 1
        self.offset = (0,0)
        self.image_id = None
        self.drawing = False
        self.lines = []

        self.bind('<Button-1>', self.start_drag)
        self.bind('<B1-Motion>', self.drag)
        self.bind('<MouseWheel>', self.zoom)
        self.bind('<Button-4>', self.zoom)
        self.bind('<Button-5>', self.zoom)
        self.bind("<ButtonPress-3>", self.start_drawing)
        self.bind("<ButtonRelease-3>", self.stop_drawing)
        self.bind("<B3-Motion>", self.draw)

    def load_array(self, arr):

        self.delete('all')
        self.scale_factor:int  = 1
        self.offset = (0,0)
        self.image_id = None
        if arr is not None:
            onechan = arr[:,:,49]
            normed_gray = (((onechan - np.min(onechan)) / (np.max(onechan) - np.min(onechan))) * 255).astype('uint8')
            normed_gray = Image.fromarray(normed_gray, mode= 'L')
            self.pil_image = normed_gray
            self.update_image()

    def start_drag(self, event):
        self.start_x, self.start_y = event.x, event.y
    
    def drag(self, event):
        if self.drawing:
            return
        dx = event.x- self.start_x
        dy = event.y- self.start_y
        self.move(self.image_id, dx, dy)

        self.start_x, self.start_y = event.x, event.y

        # update offset
        offset_x = self.offset[0]+ dx
        offset_y = self.offset[1]+ dy
        self.offset = offset_x, offset_y
        self.configure(scrollregion=self.bbox("all"))

    def update_image(self):
        width = int(self.pil_image.width * self.scale_factor)
        height = int(self.pil_image.height * self.scale_factor)

        self.resized_image = self.pil_image.resize((width, height), Image.Resampling.NEAREST)    # used nearest to avoid interpolation
        self.photo_image = ImageTk.PhotoImage(self.resized_image)

        if self.image_id:
            self.delete(self.image_id)

        anchor_x, anchor_y = int(self.canv_len/2)+self.offset[0], int(self.canv_len/2)+self.offset[1]
        self.image_id = self.create_image(anchor_x, anchor_y, image=self.photo_image, anchor='center')
        self.configure(scrollregion=self.bbox("all"))           # makes the item in canvas unable to exceed border of canvas

    def zoom(self, event):
        if self.drawing:
            return

        if event.delta > 0 or event.num == 4:
            self.scale_factor += 1
        elif event.delta < 0 or event.num == 5:
            self.scale_factor= max(self.scale_factor-1, 1)

        self.update_image()

    def start_drawing(self, event):
        global draw_start_x, draw_start_y
        self.drawing = True
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

        # Apply convex hull algorithm
        hull = ConvexHull(largest_slice)
        convex_coords = [largest_slice[i] for i in hull.vertices]

        image = Image.new("RGBA", (self.winfo_width(), self.winfo_height()), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.polygon(convex_coords, fill=(128, 0, 0, 128))

        self.drawed = ImageTk.PhotoImage(image)
        anchor_x, anchor_y = int(self.canv_len/2)+self.offset[0], int(self.canv_len/2)+self.offset[1]
        self.create_image(anchor_x, anchor_y, image=self.drawed, anchor='center')

        # self.resized_image = draw
        # self.update_image()

canvas = ZoomDrag(window)
open_button = tk.Button(window, text="Open Image", command= open_hsi)

open_button.grid(row= 0, pady= (5, 0))
canvas.grid(row= 1, padx= 5, pady= 5)

# Start the Tkinter event loop
window.mainloop()