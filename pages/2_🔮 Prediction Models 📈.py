# import module

import streamlit as st

# Title
st.title("Hello Class.... Welcome to CMP7005 - Programming for Data Analysis !!!")

# Header
st.header("This is a header")

# Subheader
st.subheader("This is a subheader")

# Text
st.text("Hello Class!!!")

# Markdown:
#-----------

# Markdown
st.markdown("# This is a markdown") # refer to this for more information on markdown - https://www.markdownguide.org/basic-syntax/

################################################
# Display: Success, Info, Warning, Error, Exception:
#################################################
# success
st.success("Success")

# success
st.info("Information")

# success
st.warning("Warning")

# success
st.error("Error")

# Exception - This has been added later
exp = ZeroDivisionError("Trying to divide by Zero")
st.exception(exp)

st.exception(RuntimeError("RuntimeError exception"))

###############################
#Display Images:
#################################

# import Image from pillow to open images

# from PIL import Image
# img = Image.open("space (1).jpg")

# # display image using streamlit
# # width is used to set the width of an image
# st.image(img, width=800)
#######################################################################################################################################
#Input widgets
##########################################
#Using write function, we can also display code in coding format. This is not possible using st.text(”).

st.write("This is CMP7005- Programming for data analysis class")

# Writing python inbuilt function range()
st.write(range(10))

#Checkbox:
#---------
#A checkbox returns a boolean value . When the box is checked, it returns a True value else returns a False value.

# check if the checkbox is checked
# title of the checkbox is 'Show/Hide'
if st.checkbox("Show/Hide"):

    # display the text if the checkbox returns True value
    st.text("Welcome to Programming for Data Analysis- CMP7005")

#Radio Button:
#------------
# first argument is the title of the radio button
# second argument is the options for the radio button
status = st.radio("Select Gender: ", ('Male', 'Female'))

# conditional statement to print
# Male if male is selected else print female
# show the result using the success function
if (status == 'Male'):
    st.success("Male")
else:
    st.success("Female")

# Selection Box:
#--------------
#You can select any one option from the select box.

# first argument takes the title of the selection box
# second argument takes options
hobby = st.selectbox("Hobbies: ",
                     ['Dancing', 'Reading', 'Sports'])

# print the selected hobby
st.write("Your hobby is: ", hobby)

#Multi-Selectbox:
#---------------

#The multi-select box returns the output in the form of a list. You can select multiple options.

# first argument takes the box title
# second argument takes the options to show
hobbies = st.multiselect("Hobbies: ",
                         ['Dancing', 'Reading', 'Sports'])

# write the selected options
st.write("You selected", len(hobbies), 'hobbies')

# Button:
#-------

#st.button() returns a boolean value . It returns a True value when clicked else returns False .

# Create a simple button that does nothing
st.button("Click me for no reason")

# Create a button, that when clicked, shows a text
if(st.button("About")):
    st.text("Welcome to Programming for Data Analysis- CMP7005!!!")

# Text Input:
#-----------

# save the input text in the variable 'name'
# first argument shows the title of the text input box
# second argument displays a default text inside the text input area
name = st.text_input("Enter Your name", "Type Here ...")

# display the name when the submit button is clicked
# .title() is used to get the input text string
if(st.button('Submit')):
    result = name.title()
    st.success(result)

# slider:
#-------
# first argument takes the title of the slider
# second argument takes the starting of the slider
# last argument takes the end number
level = st.slider("Select the level", 1, 5)

# print the level
# format() is used to print value
# of a variable at a specific position
st.text('Selected: {}'.format(level))

'''
st.number_input(): This function is used to display a numeric input widget.
st.text_input(): This function is used to display a text input widget.
st.date_input(): This function is used to display a date input widget to choose a date.
st.time_input(): This function is used to display a time input widget to choose a time.
st.text_area(): This function is used to display a text input widget with more than a line text.
st.file_uploader(): This function is used to display a file uploader widget.
st.color_picker(): This function is used to display color picker widget to choose a color.
'''

st.number_input('Pick a number', 0, 10)
st.text_input('Email address')
st.date_input('Traveling date')
st.time_input('School time')
st.text_area('Description')
st.color_picker('Choose your favorite color')

##############################################################################################
#Display status with Streamlit
###################################################
'''
Sidebar and container
You can also create a sidebar or a container on your page to organize your app.
The hierarchy and arrangement of pages on your app can have a large impact on your user experience.
By organizing your content, you allow visitors to understand and navigate your site,
which helps them find what they're looking for and increases the likelihood that they'll return in the future.
'''''

st.sidebar.title("This is the Navigation Sidebar")
st.sidebar.button('Click')
st.sidebar.radio('select',['Data','EDA','Model Building'])
st.sidebar.file_uploader('Upload your file')
