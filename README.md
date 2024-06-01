## Project description

The main goal of this project is to create a model, which can recognise the location (in our case city) based on a Google Streetview image taken in that city. To do this for the whole world this would require an enormous 
dataset and very long training times. Therefore we restricted ourselves to images from 23 distinct cities as they are given in this [Kaggle dataset](https://www.kaggle.com/datasets/amaralibey/gsv-cities). 

The dataset contains around 500,000 streetview images that coming from 23 different cities. Some locations appear multiple times in the dataset but the photos were taken at different point in times, which reduces the dataset to around 60,000 
unique locations. 

This project was inspired by the game Geoguessr, where players are randomly placed at a Google streetview location and then have to find our/guess where they are. Successful players are using various hints to determine the 
country or region that they are currently in such as street signs, license plate colours, road markings etc. Our model aims to make use of such features by trying to identify whether for example a street sign is present in
a given picture and if so determine the region of that feature. We then combine these models along with a plain neural network model to obtain a final guess of the city that the picture was taken in.

## Key stakeholders

Groups that may be interested in such technologies are:
- Police and government securiuty agencies
- Professional Geoguessr players (model explainability could give rise to new useful hints for players)
- Investigative journalists

## Key performance indicators

Prior to training we split the dataset into a training set consisting of 80% of the data, and a test set consisting of the remaining 20% of the data. The key metric will be accuracy (that is, the percentage of correctly predicted examples).

## Methods

For training we used performed a random 80-20 split of the aforemenetioned dataset stratified along the classes, and made sure to have no overlap in places between the training and test set. The split dataset can be found [here](https://www.kaggle.com/datasets/bezemekz/gsv-cities-cleaned-normalized-train-test).


## Results

## Installation requirements
The necessary packages and their versions can be found in `requirements.txt`. It is possible to install these via `pip install -r requirements.txt` once the repository was cloned and pulled.
