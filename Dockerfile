FROM continuumio/miniconda3
RUN conda install numpy scipy scikit-learn pandas joblib
RUN pip install deap update_checker tqdm stopit xgboost tpot
COPY main.py main.py
COPY bank.csv bank.csv
CMD [ "python", "main.py" ]