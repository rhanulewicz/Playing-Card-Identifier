# Playing-Card-Identifier
I provide the card identification code ('code/main.py') as well as the template data needed to make it work ('data/cards_png/'). 

I also provide a folder with a piece of of input data to try out ('data/input/'), just in case you don't feel like taking your own photographs of playing cards. You can if you want, though!

The only parameters meant for the user to change are the threshold value:

Line 8: THRESHOLD = 128

and input image path:

Line 9: img = cv2.imread('../data/input/3c.jpg', cv2.IMREAD_GRAYSCALE)

I only recommend changing the threshold value if the default one is not working for you. Feel free to play around with it though!
