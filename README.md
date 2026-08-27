# NeuroBiomark
Univeristy project focused on using DenseNet121 model with attention mechanism techniques as a means for pathology prediction in ALS patients


# Navigation
- GUI: version of the model with a graphical user interface. Can be used to infere images one-by-one.
- Trainable models: contains code for train loops and augmentation evaluation.
-   3 Classes: Original trainable model, contains Concordant, Discordant and Control classes
-   2 Classes: Modified trainable model, contain Concordant and Discordant classes, as well as different augmentation techniques.

# Requirements
## Packages
The GUI model has requirements established by the people who created it. The trainable models however, do not. As such, I developed a workaround of applying the requirements from the GUI model to the trainable ones. There are some problems with this approach. Firstly, you still have to install 'seaborn' and 'albumentations' packages. Secondly, each time the model reads an image file of the cells it gives a warning since it misses some c++ library for metadata recognition. And finally, you have to reinstall torch for cuda, as it'll be installed fro cpu.

The requirements for the GUI model are:
> IMPORTANT - MAKE SURE YOU HAVE PYTHON VERSION 3.9.7 INSTALLED AND SELECT IT AS YOUR PYTHON INTERPRETER FOR YOUR VIRTUAL ENVIRONMENT BEFORE YOU DO THE FOLLOWING.
> If any changes are made to the repo, make sure to close and rerun gui_home.py to see the changes

> 1. Create a virtual enviroment - python3 -m venv venv
> 2. Activate that virtual enviornment - For macOS, "source venv/bin/activate", and for Windows, ".\venv\Scripts\Activate"
> 3. Install all of the necessary dependancies from the requirements.txt file - pip install -r requirements.txt
> 4. cd into the root directory of the project first "NeuroBiomark_Project_interface"
> 5. Run the application file - python -m Home_Window.gui_home ```

## Miscellaneous files 
Every 'dataset' folder will be missing 'image_kayes.xlsx' as it contains sensitive information not to be published online. Nonetheless, having this file is necessary for trainable models to create folds. As such, you will need to manually add it to the directory.

> (this applies only for the trainable models)
> Additionally, the paths written in the 'config.py' folders, which are referenced everywhere in the code, are absolute. Therefore, you'll need to update them after cloning the project fro the model's to run.


# TODO
- [ ] Trainable models contain code for EfficientNet and BasicCNN which aren't actually being used, this clutters the project and makes it hard to understand. Better remove them.
- [ ] Big chunk of the project is may have been written sequentially by AI, it would be a good idea to tidy it up into more manageable files, as well as remove unnecessary duplicated functions.
- [ ] Add MCC evaluation for training & augmentation techniques evaluation.
- [x] Run the model with 2 classes with all the good augmentations
-   Result: The model did not get better, looking at the graphs of the training and validation losses changes through epochs, it seems that the model suffers from overfitting as well as unstable learning issues.
- [ ] Plot Confidence level of prediction in function of ECAS/ALSFRS-R value.
- [ ] Add Train & Val losses over epochs graphs for training to pinpoint problems with the models more accurately.
- [ ] Change config.py paths from absolute to relative
  
