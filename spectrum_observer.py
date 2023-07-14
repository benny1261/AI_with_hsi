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
    def __init__(self, master: any, width:int= 1000, height:int= 600, bg = 'black', **kwargs):
        super().__init__(master, width=width, height=height, bg= bg, **kwargs)
        self.scale_factor:int  = 1
        self.offset = (0,0)
        self.arr = None
        self.image_id = None
        self.mask_id = None
        self.drawing = False
        self.lines = []
        self.mask_color = (255,0,0,40)

        self.bind('<ButtonPress-1>', self.start_drag)
        self.bind('<B1-Motion>', self.drag)
        self.bind('<ButtonRelease-1>', self.stop_drag)
        self.bind('<MouseWheel>', self.zoom)
        self.bind('<Button-4>', self.zoom)
        self.bind('<Button-5>', self.zoom)
        self.bind("<ButtonPress-3>", self.start_drawing)
        self.bind("<ButtonRelease-3>", self.stop_drawing)
        self.bind("<B3-Motion>", self.draw)

    def load_array(self, arr):
        self.delete('all')
        self.arr = arr
        self.scale_factor:int  = 1
        self.offset = (0,0)
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
        self.offset = offset_x, offset_y

    def update_image(self):
        width = int(self.pil_image.width * self.scale_factor)
        height = int(self.pil_image.height * self.scale_factor)

        self.resized_image = self.pil_image.resize((width, height), Image.Resampling.NEAREST)    # used nearest to avoid interpolation
        self.photo_image = ImageTk.PhotoImage(self.resized_image)

        anchor_x, anchor_y = int(self.winfo_width()/2)+self.offset[0], int(self.winfo_height()/2)+self.offset[1]
        self.image_id = self.create_image(anchor_x, anchor_y, image=self.photo_image, anchor='center', tags= ('image'))
        if self.mask_id:
            self.tkmask = ImageTk.PhotoImage(self.mask.resize((self.mask.width*self.scale_factor,self.mask.height*self.scale_factor),
                                                              Image.Resampling.NEAREST))
            self.mask_id = self.create_image(anchor_x, anchor_y, image=self.tkmask, anchor= 'center', tags= ('image'))

    def zoom(self, event):
        if self.drawing or not self.image_id:
            return

        if event.delta > 0 or event.num == 4:
            self.scale_factor += 1
        elif event.delta < 0 or event.num == 5:
            self.scale_factor= max(self.scale_factor-1, 1)

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

        if self.scale_factor != 1:
            self.mask = self.mask.resize((self.pil_image.width, self.pil_image.height), Image.Resampling.NEAREST)
            self.tkmask = ImageTk.PhotoImage(self.mask.resize((self.mask.width*self.scale_factor,self.mask.height*self.scale_factor),
                                                              Image.Resampling.NEAREST))
        else:
            self.tkmask = ImageTk.PhotoImage(self.mask)
        self.mask_id = self.create_image(bbox[0], bbox[1], image=self.tkmask, anchor=tk.NW, tags= ('image'))

canvas = ZoomDrag(window)
open_button = tk.Button(window, text="Open Image", command= open_hsi)

open_button.grid(row= 0, pady= (5, 0))
canvas.grid(row= 1, padx= 5, pady= 5)

# Start the Tkinter event loop
window.mainloop()