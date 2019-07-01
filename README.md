# AuthorIdentification

Simple but effective classification model for predicting the author of a piece of text.
The specific model in this repo is based on wine reviews by 20 different authors.

## Usage
### Author Prediction

First you instantiate the model. This builds and serializes it, if it has not previously done so, otherwise, it simply loads it from disk.

```python
from wine_model import WineModel
wm = WineModel()
```

Then you can apply the model using the `predict_review_author()` method which takes as input, a list of string texts for which you want to know the authors. Please note that even if you only want to predict for one author, it must still be in a list.
```python
wm.predict_review_author(["Crushed thyme, alpine wildflower, beeswax and orchard-fruit aromas are front and center on this beautiful white. Proving just how well Pinot Bianco can do in Alto Adige, the creamy palate is loaded with finesse, delivering ripe yellow pear, creamy apple and citrus alongside tangy acidity. White-almond and stony mineral notes back up the finish..."])
```
which gives 
```python
{"confidence":0.901233228, "prediction":"Kerin OKeefe"}
```

### Hyperparameter Optimization
A script is also provided to aid in selecting model hyperparameters using Particle Swarm Optimization (PSO). This can be run simply with `python pso_hyperparam_opt.py` but also has the optional argument `--vratio` which is a float that controls what proportion of the dataset should be used as the validation set.
