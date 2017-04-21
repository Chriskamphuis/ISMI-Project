### ISMI-Project                                                                  
Repository for the [Cervix Cancer screening
competition](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening)
for the course Intelligent Systems in Medical Imaging.


##### Use the following structure for specefic parts of the project:                  

* ISMI-Project/data/raw/ - for the raw data 
* ISMI-Project/data/pre/ - for the preprocessed data  
* ISMI-Project/scripts/ - for python scripts (like the data generators etc.)                   
* ISMI-Project/notebooks/ - for the notebooks

**Do not push any data to the repo!**

General ideas/tips: (add more if you have any ( especially you Jonas ;-) )

* Try to write code that is easy to plugin, for example the ideal data augment
generator is just a wrapper around another generator. This way you can just
choose wheter you want to augment data or not.
* Try to make stuff _modular_, for the example of the data augmenter. Try to
make it so that you can easily choose what augments you want to use and which
ones you do not want to use with use of parameters. This way you can easily
compare different types of augments with only a parameter change.
* When creating function that are core for the project, try to write some clear
discription at the top of the function. This way other people can pick them
up with more ease. 
* Make issues for things you want to try, this way people know what you are
doing and we have a better overview of what everyone is doing. 
* When you write code that is important for the project, also create an issue
for testing the code. Then someone else can test your code and see if it
works properly.
