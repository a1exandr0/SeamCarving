from SeamCarving.module.SeamCarve import SeamCarve
import tkinter as tk
from PIL import Image as IM
from PIL import ImageTk as IMTK
import numpy as np
"""
Interface to simplify usage of SeamCarving class.
You mostly need to use interface when you want to use protective or targeting masking in order to remove or keep certain
objects after retageting.

INSTRUCTION:
1)Run Interface.py
2)Type image name into the console
3)IF you want specific areas to be masked, choose between protect and target and press that button
    3.1)Click all the areas you want to be masked
    3.2)Press "apply weights button"
    NOTE:Cant apply different masks at the same time(You cant protect 1 object and target another at the same time)
4)Enter scaling proportion on the bottom of frame
5)Press "reshape" 

"""

MODE = ""
WEIGHTS = []
IMG = SeamCarve()
MASK_ADD = 10**6
MASK_DEC = -(10**6)
SCALE = 1
circle = 0
ENERGY_COPY = None


def motion(event):
    """
    Adds circle around mouse pointer in order to make user understand what area was chosen.
    :param event:
    :return:
    """
    x, y = event.x, event.y

    global circle
    global canvas

    canvas.delete(circle)  #to refresh the circle each motion

    radius = 20  #change this for the size of your circle

    x_max = x + radius
    x_min = x - radius
    y_max = y + radius
    y_min = y - radius

    circle = canvas.create_oval(x_max, y_max, x_min, y_min, outline="red")


def on_click(event=None):
    """
    Adds all dots of circle to WEIGHTS after you click mouse button.
    :param event:
    :return:
    """
    if MODE == "":
        pass
    else:
        off = IMG._image.shape
        x = img.winfo_pointerx() - img.winfo_rootx()
        y = img.winfo_pointery() - img.winfo_rooty()
        # y = off[0] - y
        print("X:{}, Y:{}".format(x, y))
        if x <= 20:
            x_min_lim = 0
        else:
            x_min_lim = x - 20
        if y <= 20:
            y_min_lim = 0
        else:
            y_min_lim = y - 20
        if off[1] - x <= 20:
            x_max_lim = off[1]
        else:
            x_max_lim = x + 20
        if off[0] - y <= 20:
            y_max_lim = off[0]
        else:
            y_max_lim = y + 20

        for X in range(x_min_lim, x_max_lim):
            for Y in range(y_min_lim, y_max_lim):
                if (X - x)**2 + (Y - y)**2 < 20**2:
                    WEIGHTS.append([X, Y])


def kill(event=None):
    """
    Stops the Interface and closes it.
    :param event:
    :return:
    """
    img.destroy()
    control_panel.destroy()


def reset_weights(event=None):
    """
    Resets WEIGHTS list and self._mask.
    :param event:
    :return:
    """
    global WEIGHTS, IMG
    WEIGHTS = []
    IMG.energy_map_w_filter()


def apply_weights(event=None):
    """
    For each element of WEIGHTS changes self._mask and self._energy_map.
    :param event:
    :return:
    """
    global WEIGHTS
    IMG.energy_map_w_filter()
    for i in WEIGHTS:
        IMG._mask[i[1]][i[0]] = 1
        if MODE == "protect":
            IMG._energy_map[i[1]][i[0]] = MASK_ADD
        elif MODE == "target":
            IMG._energy_map[i[1]][i[0]] = MASK_DEC

    WEIGHTS = []


def add_weights(event=None):
    """
    Turns to protect mode.
    :param event:
    :return:
    """
    global MODE
    global WEIGHTS
    MODE = "protect"
    WEIGHTS = []
    l1.configure(text="protect")


def dec_weights(event=None):
    """
    Turns to targeting mode.
    :param event:
    :return:
    """
    global MODE
    global WEIGHTS
    MODE = "target"
    WEIGHTS = []
    l1.configure(text="target")


def retarget(event=None):
    """
    Reshapes the image and saves the result in out.png
    :param event:
    :return:
    """
    global SCALE
    SCALE = float(e.get())

    if SCALE == 1:
        pass
    elif SCALE > 1:
        IMG.scale_up(SCALE)
    elif SCALE < 1:
        IMG.scale_down(SCALE, MODE)
    IMG.build("out.png")
    l1.configure(text="Image saved in out.png")
    im = IMTK.PhotoImage(IM.open("out.png"))
    canvas.delete("all")
    off = IMG._image.shape
    canvas.configure(width=off[1], height=off[0])
    canvas.create_image(int(off[1]/2), int(off[0]/2), image=im)
    canvas.image = im
    IMG._mask = np.zeros(IMG._image.shape[:2], dtype=np.bool)


image_name = input("set image name here")
img = tk.Tk()
control_panel = tk.Tk()
control_panel.geometry("+3000+3000")
img.resizable(False, False)
control_panel.resizable(False, False)

img.bind("<Motion>", motion)

IMG.fit(image_name)
off = IMG._image.shape


canvas = tk.Canvas(width=off[1], height=off[0], bg='white')
canvas.pack()


im = IMTK.PhotoImage(IM.open(image_name))
canvas.create_image(int(off[1]/2), int(off[0]/2), image=im)




img.bind('<Button-1>', on_click)



b1 = tk.Button(control_panel, text="Close", command=kill)
b1.pack()


b2 = tk.Button(control_panel, text="Reset Weights", command=reset_weights)
b2.pack()


l1 = tk.Label(control_panel, text="Mode")
l1.pack()


b3 = tk.Button(control_panel, text="Apply Weights", command=apply_weights)
b3.pack()


b4 = tk.Button(control_panel, text="Protect", command=add_weights)
b4.pack()


b5 = tk.Button(control_panel, text="Target", command=dec_weights)
b5.pack()


b6 = tk.Button(control_panel, text="Reshape", command=retarget)
b6.pack()

l2 = tk.Label(control_panel, text="Set Scaling Here")
l2.pack()


e = tk.Entry(control_panel, text="proportion here")
e.pack()


control_panel.mainloop()
img.mainloop()