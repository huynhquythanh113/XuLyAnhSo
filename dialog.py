# python program demonstrating
# Combobox widget using tkinter


from cgitb import text
import tkinter as tk
from tkinter import Button, Entry, StringVar, ttk
from typing_extensions import IntVar
import cv2
from skimage.util import random_noise
import numpy as np

image = cv2.imread("image/img7.jpg")

# Creating tkinter window
window = tk.Tk()
window.title('Combobox')
window.geometry('500x250')

# label text for title
ttk.Label(window, text="GFG Combobox Widget",
          background='green', foreground="white",
          font=("Times New Roman", 15)).grid(row=0, column=1)

# label
ttk.Label(window, text="Select the Month :",
          font=("Times New Roman", 10)).grid(column=0,
                                             row=5, padx=10, pady=25)


ttk.Label(window, text="Size of filter :",
          font=("Times New Roman", 15)).grid(row=1, column=0)

size = StringVar()

t1 = Entry(window, textvariable=size, font=(14))
t1.grid(column=1, row=1)
# Combobox creation
n = tk.StringVar()
monthchoosen = ttk.Combobox(window, width=27, textvariable=n)


def select():
    if (n.get() == 'Median'):
        median = cv2.medianBlur(image, int(size.get()))
        cv2.imshow("After medianBlur", median)
        cv2.imshow("Image original", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if (n.get() == 'Gaussian'):
        median = cv2.GaussianBlur(
            image, (int(size.get()), int(size.get())), cv2.BORDER_DEFAULT)
        cv2.imshow("After GaussianBlur", median)
        cv2.imshow("Image original", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if (n.get() == 'Mean'):
        im1 = cv2.blur(image, (int(size.get()), int(
            size.get())), cv2.BORDER_DEFAULT)
        im2 = cv2.boxFilter(image, -1, (int(size.get()),
                            int(size.get())), normalize=False)
        cv2.imshow("After Mean", np.hstack((im1, im2)))
        cv2.imshow("Image original", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if (n.get() == 'LoG'):
        median = cv2.GaussianBlur(
            image, (int(size.get()), int(size.get())), cv2.BORDER_DEFAULT)
        lap = cv2.Laplacian(median,cv2.CV_64F)
        laplacian1 = lap/lap.max()
        cv2.imshow("After Mean", laplacian1)
        cv2.imshow("Image original", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    
    # print(int(size.get())+1)
    # print(n.get())


# Button
b1 = Button(window, command=select, text='Select', font=(14))
b1.grid(column=1, row=6)

# Adding combobox drop down list
monthchoosen['values'] = ('Mean',
                          'Gaussian',
                          'Median',
                          'LoG')

monthchoosen.grid(column=1, row=5)
monthchoosen.current()
window.mainloop()
