# NeuroBiomark
Univeristy project focused on using DenseNet121 model with attention mechanism techniques as a means for pathology prediction in ALS patients


# Navigation
- GUI: version of the model with a graphical user interface. Can be used to infere images one-by-one.
- Trainable models: contains code for train loops and augmentation evaluation.
-   3 Classes: Original trainable model, contains Concordant, Discordant and Control classes
-   2 Classes: Modified trainable model, contain Concordant and Discordant classes, as well as different augmentation techniques.

# Requirements
The GUI model has requirements established by the people who created it. The trainable models however, do not. As such, I developed a workaround of applying the requirements from the GUI model to the trainable ones. There are some problems with this approach. Firstly, you still have to install 'seaborn' and 'albumentations' packages. Secondly, each time the model reads an image file of the cells it gives a warning since it misses some c++ library for metadata recognition. And finally, you have to reinstall torch for cuda, as it'll be installed fro cpu.

The requirements for the GUI model are:
> IMPORTANT - MAKE SURE YOU HAVE PYTHON VERSION 3.9.7 INSTALLED AND SELECT IT AS YOUR PYTHON INTERPRETER FOR YOUR VIRTUAL ENVIRONMENT BEFORE YOU DO THE FOLLOWING.
> If any changes are made to the repo, make sure to close and rerun gui_home.py to see the changes

> 1. Create a virtual enviroment - python3 -m venv venv
> 2. Activate that virtual enviornment - For macOS, "source venv/bin/activate", and for Windows, ".\venv\Scripts\Activate"
> 3. Install all of the necessary dependancies from the requirements.txt file - pip install -r requirements.txt
> 4. cd into the root directory of the project first "NeuroBiomark_Project_interface"
> 5. Run the application file - python -m Home_Window.gui_home ```


# TODO
- [ ] Trainable models contain code for EfficientNet and BasicCNN, it clutters the project and makes it hard to understand what functions to what. Better remove them.
- [ ] Big chunk of the project is written sequentially by AI, it would be a good idea to tidy up the code into more manageable files, as well as remove unnecessary duplicated functions.
- [ ] Add MCC evaluation for augmentation techniques
- [ ] Run the model with 2 classes with all the good augmentations
- [ ] Plot Confidence level of prediction in function of ECAS/ALSFRS-R value.
  
