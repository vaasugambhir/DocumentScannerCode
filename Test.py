# a sample file to see how the scanner module works

import cv2
import os
import scanner as scan

# type any image name here
img = cv2.imread('doc11.jpg')

data = scan.convert_numpy_to_base64_array(img)
image = scan.convert_base64_array_to_numpy(data)


# to get the scanned image and other images (its binary output, etc)
scanned_image = scan.get_scanned_image_auto(data)

# you can set any path here to a folder
os.chdir('c:/Users/hp-2111/Desktop/Scanner spider/final images/')

# if you want to show the images directly
# scan.show_images(images, names)

# if you want to save images
scan.save_images(images, names)

cv2.imshow('scan.jpg', scanned_image)
cv2.imshow('scanned binary.jpg', binary_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# I suggest you use a different folder in a similar path from here
os.chdir('c:/Users/hp-2111/Desktop/Scanner spider/final features/')

# changing brightness example
new1 = scan.change_brightness_contrast(scanned_image, 0, 1.25)
cv2.imwrite('contrast change.jpg', new1)
new1 = scan.change_brightness_contrast(scanned_image, 50, 1)
cv2.imwrite('brightness change.jpg', new1)

# changing HSV example
new1 = scan.change_HSV(scanned_image, h_offset=20, s_offset=0, v_offset=0)
cv2.imwrite('hsv change1.jpg', new1)
new1 = scan.change_HSV(scanned_image, h_offset=0, s_offset=0, v_offset=30)
cv2.imwrite('hsv change2.jpg', new1)

# changing RGB example
new1 = scan.change_RGB(scanned_image, r_offset=20, g_offset=0, b_offset=0)
cv2.imwrite('red change.jpg', new1)
new1 = scan.change_RGB(scanned_image, r_offset=0, g_offset=20, b_offset=0)
cv2.imwrite('green change.jpg', new1)
new1 = scan.change_RGB(scanned_image, r_offset=0, g_offset=20, b_offset=20)
cv2.imwrite('blue change.jpg', new1)

# resizing example
new1 = scan.change_size(scanned_image, 1)
cv2.imwrite('size change low.jpg', new1)
new1 = scan.change_size(scanned_image, 2)
cv2.imwrite('size change medium.jpg', new1)
new1 = scan.change_size(scanned_image, 3)
cv2.imwrite('size change optimal.jpg', new1)
cv2.imwrite('scanned image.jpg', scanned_image)
