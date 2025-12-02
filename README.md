
This Repository contains the colab notebook for the hybrid framework described in the abstract below:


In modern data-driven intralogistics, Automated Guided Vehicles (AGVs) play a central role. Mission success is identified with the positive conclusion of an operation - the loading or the unloading of an object in the facility. Although this binary measure can be useful in fault prevention, it is lacking to capture more subtle problems such as congestion, delays in the start of a missions et cetera. 

Our focus is towards the definition of a framework capable of finding these subtleties; such framework is in need of a model capable of capturing such complexities. The main issue within this regard is the way data are stored in mission logs: mainly categorical data, very sparse and with much useless information. To solve this, we decided to reduce our observations to as few variables as possible -- start and end nodes, operation type, mission duration, and time delay between the mission assignment time and the actual mission starting time.

We propose a hybrid autoencoder-SPC framework that can provide interpretable and industrially relevant monitoring of AGV fleets.
