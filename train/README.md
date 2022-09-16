# Classification Tuner
Experimental setup for hyperparameter tuning for code smells classification 

## Requirements

1. Python 3 (developed with `Python 3.7.x`)
2. Optional: virtualenv for virtual environment: `pip install virtualenv`

## Setup 

1. Clone the repository and move into folder
2. If using virtual environment:
  - create virtual environment: `virtualenv -p python3 env`
  - activate virtual environment: `source env/bin/activate`  
3. Install requirements: `pip install -r requirements.txt`

## Scripts

* `auto_model.py` - HyperModel for Bayesian Optimization, currently based on RandomForest, SVM (via Bagging), Catboost and XGBClassifier combined with sampling strategies. 
  Parameter spaces can be (re)defined here.  
* `classifier.py` - main script with tuner definition, data handling, hyperparameter tuning, and metrics calculation.  

## Execution  

* `python classifier.py` 
* **Note:** setting of the sampling strategy can be done via passing `sampling`
parameter in the constructor of `AutoModel` class:
  - `sampling=None` or no passing parameter at all - no sampling (default behavior)
  - `sampling="over"` - over sampling 
  - `sampling="under"` - under sampling
  - `sampling="combine"` - combine sampling
    