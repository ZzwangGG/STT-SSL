## ST-SSL: Spatio-Temporal Self-Supervised Learning for Traffic Prediction 

Specifically, we design a spatiotemporal Transformermodule that leveragesamulti-headattentionmechanismtoaddress the
limitationsoftraditionalsequentialmodelsinhandlinglong-term memory. Thismodulemodels spatial dependencies fromboth
fixedanddynamicperspectives,allowingcomprehensivecapture ofcomplexspatial features.Toovercomethe incompletefeature
representationof trafficdataandenhancemodelrobustness,we implement adaptivegraphaugmentationtodynamicallyadjust
the graph structure frommultipledimensions. Toaddress the spatiotemporalheterogeneitypresent infine-grainedtrafficdata,
we introduce a spatiotemporal self-supervisedmodule, which employs twoauxiliaryself-supervisedlearningtasks to improve
the model’s ability to capture spatiotemporal heterogeneity.

## Requirement

We build this project by Python 3.8 with the following packages: 
```
numpy==1.21.2
pandas==1.3.5
PyYAML==6.0
torch==1.10.1
...
```

## Model training and Evaluation

If the environment is ready, please run the following commands to train model on the specific dataset from `{PeMS04/PeMS08/PeMS-BAY/METR-LA}`.
```bash
>> cd STT-SSL
>> ./runme 0 PEMS08-12   # 0 gives the gpu id
```


