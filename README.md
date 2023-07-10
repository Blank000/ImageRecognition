## Project Description - 

In this project, it's detecting different objects in an Image based on pre-trained model.
We are making use of `RetinaNet` model from Keras which is initialised with `retinanet_resnet50_pascalvoc` preset. We are then loading images, resizing it to an absolute value of ( `640 X 640` ), create mapping for class objects, then generates prediction. Finally there is a visualisation function to create visualisation for the bounding box in the image ( `visualization.plot_bounding_box_gallery()` )

## Pre-requisites -

* python installed >= 3.10

* Clone this project on local
    ```
    git clone https://github.com/Blank000/ImageRecognition.git
    ```
    or download and unzip from 
    
    https://github.com/Blank000/ImageRecognition/archive/refs/heads/main.zip

## Project Setup -

#### Create Python Virtual Environment

Before running this move to the <PROJECT_DIR> where the project is located.

    > cd <PROJECT_DIR>
   
    Create envrionments as below - 

    * Using virtualenv - 

        > virtualenv env

        > source <PROJECT_DIR>/<ENV_NAME>/bin/activate

    * Using python -m -

        > python -m venv env
        
        > source <PROJECT_DIR>/<ENV_NAME>/bin/activate

#### setup.sh

If on Linux / Ubuntu / macOS operating system please run the setup.sh

```
sh <PROJCT_DIR>/setup.sh
```

If present on windows or other operating system, please execute the commmands inside `setup.sh`

#### Run ImageRecognition

If you want to change images, please change it in main.py:16
python <PROJECT_DIR>/main.py
