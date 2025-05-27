LaptopPricePrediction/
│
├── data/
│   └── laptops.csv                # Your main dataset
│
├── distributions/
│   └── feature_distributions.png  # Plots or visuals
│
├── python/
│   ├── data_preprocessing.py     # Preprocessing steps
│   ├── model_training.py         # Training multiple models & evaluation
│   ├── best_model_saver.py       # Save the best model as .pkl
│   ├── predict_sample.py         # Load model & make prediction on new sample
│   └── config.py                 # Constants, file paths
│
├── best_model.pkl                # Saved best model (SVR)
└── README.md                     # Project description & instructions
